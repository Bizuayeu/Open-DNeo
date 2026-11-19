//! Open-DNeo's oracle, kept in a local tree only: one stream's rounds as the lane core takes them under GLM's lone
//! DFlash2 facts, written as JSON lines for open_dneo.rounds.served to walk the same reply.
const std = @import("std");
const cfgmod = @import("config.zig");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const be = @import("backend.zig");
const fake = @import("fake.zig");
const ev = @import("events.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;

const a = std.testing.allocator;
const out_path = "open-dneo-oracle.jsonl";
const cut = 0.7; // TF_GLM_DRAFT_CONFIDENCE's default (families/glm/dflash.zig)
const block = 7; // a DFlash2 block's drafts (cuda_lanes.zig max_held)

fn costTable(comptime n: usize, comptime one: f64, comptime row: f64) [n]cfgmod.Cost {
    var c: [n]cfgmod.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = one + row * @as(f64, @floatFromInt(i)) };
    return c;
}

// families/glm/cuda_copy_policy.zig's two tables, which a block drafter's facts take
const window_costs = costTable(32, 52.0, 6.8);
const shared_costs = costTable(32, 76.0, 4.1);

/// families/glm/cuda_lanes.zig `facts` with a drafter and one stream (e3fb56cb09)
const glm: cfgmod.Model = .{
    .allocate_shared_drafts = false,
    .exact_width = 32,
    .first_copy_rows = 4,
    .mtp = true,
    .speculate = true,
    .speculate_early = false,
    .drafts = block,
    .draft_streams = true,
    .mtp_step_ms = 1.6 / @as(f64, block),
    .shared_step_ms = null,
    .draft_probabilities = true,
    .draft_confidence = null,
    .window_costs = &window_costs,
    .shared_costs = &shared_costs,
    .window_costs_lone = false,
    .shared_drafts = 0,
    .floor_rates = false,
    .hidden_rows = false,
    .batch_rows = 0,
    .max_streams = 1,
};

const Asked = struct { drafts: [block]u32, chances: [block]f64 };

/// The scene's state: the backend's ptr is the fake, so the wrapped calls find it here.
var state: struct {
    target: *fake.Fake = undefined,
    inner: *const fn (*anyopaque, []const be.DraftRequest) anyerror!void = undefined,
    prompt_len: u64 = 0,
    seed: u64 = 0,
    low_first: bool = false,
    sure_first: bool = false,
    held: [block]f64 = undefined, // the held drafts' chances after the cut, as draftBlock leaves them
    held_n: usize = 0,
    asked: std.AutoArrayHashMapUnmanaged(u64, Asked) = .empty,
    hook_calls: usize = 0,
    cut_binds: usize = 0, // asks whose cut held fewer drafts than the depth asked
} = .{};

/// A draft's chance among a block's 16 candidates: k / 1024, at least 0.3 (1/16 for a low first draft).
fn chance(at: u64, j: usize) f64 {
    if (state.low_first and j == 0 and at % 3 == 0) return 64.0 / 1024.0;
    if (state.sure_first and at == 0) return 1.0;
    const key: [2]u64 = .{ at, j };
    const h = std.hash.Wyhash.hash(state.seed, std.mem.asBytes(&key));
    const u: f64 = @as(f64, @floatFromInt(h % 1000)) / 1000.0;
    return (1024.0 - @floor(717.0 * u * u * u)) / 1024.0;
}

/// The fake's drafts for a whole block, then held to the depth asked, with chances cut as draftBlock cuts them.
fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    if (requests.len != 1) return error.NotLone;
    const r = requests[0];
    if (r.depth > block) return error.DeeperThanBlock;
    state.held_n = r.depth;
    if (r.depth == 0) return state.inner(ptr, requests);
    var whole = r;
    whole.depth = block;
    try state.inner(ptr, &.{whole});
    const lane = state.target.lanes.getPtr(r.stream).?;
    var got: Asked = undefined;
    @memcpy(&got.drafts, lane.held.items[0..block]);
    const at = r.position - state.prompt_len - 1;
    for (&got.chances, 0..) |*c, j| c.* = chance(at, j);
    try state.asked.put(a, at, got);
    lane.held.shrinkRetainingCapacity(r.depth);
    var chain: f64 = 1; // past the first, a chain stops where its product falls under the cut
    var kept: usize = 0;
    for (state.held[0..r.depth], got.chances[0..r.depth], 0..) |*c, p, j| {
        c.* = p;
        chain *= c.*;
        if (j > 0 and chain < cut) c.* = 0;
        if (c.* == 0) chain = 0;
        kept += @intFromBool(c.* > 0);
    }
    state.cut_binds += @intFromBool(kept < r.depth);
}

/// cuda_lanes.zig `chancesFn`: each held draft's chance of landing, the cut chances chained.
fn chances(_: *anyopaque, _: *sm.Stream, out: []f64) anyerror!bool {
    if (state.held_n == 0) return false;
    state.hook_calls += 1;
    const n = @min(out.len, state.held_n);
    var chain: f64 = 1;
    for (out[0..n], state.held[0..n]) |*c, p| {
        chain *= p;
        c.* = chain;
    }
    for (out[n..]) |*c| c.* = 0;
    return true;
}

/// Round times by the last verify's rows: a lone window's forward (cuda_copy_policy's window table) and a block's pass.
const Clock = struct {
    target: *const fake.Fake,

    fn clock(c: *Clock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *Clock = @ptrCast(@alignCast(ptr));
        const rows: f64 = @floatFromInt(@max(1, c.target.last_rows));
        return 52.0 + 6.8 * (rows - 1) + 1.6;
    }
};

const Scene = struct {
    name: []const u8,
    prompt: []const u32,
    pattern: []const u32, // the target's reply from its first token, as the fake cycles it
    max_new: u32,
    eos: []const u32 = &.{},
    seed: u64,
    low_first: bool = false,
    sure_first: bool = false, // the first round's chances all 1, so its depth shows
};

fn field(event: []const ev.Field, k: []const u8) ?ev.Value {
    for (event) |x| if (std.mem.eql(u8, x.k, k)) return x.v;
    return null;
}

fn isStr(v: ?ev.Value, s: []const u8) bool {
    const x = v orelse return false;
    return x == .str and std.mem.eql(u8, x.str, s);
}

fn list(w: *std.Io.Writer, xs: []const u32) !void {
    try w.writeByte('[');
    for (xs, 0..) |x, i| {
        if (i > 0) try w.writeByte(',');
        try w.print("{d}", .{x});
    }
    try w.writeByte(']');
}

/// One scene's stream, alone, to its end; its JSON line. Returns whether the scene's own checks held.
fn run(scene: Scene, w: *std.Io.Writer) !bool {
    var cfg = try cfgmod.Config.init(a, glm, 32, 31);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a, .cycle_after = 0, .probe_prompt = scene.prompt.len, .pattern = scene.pattern };
    defer target.deinit();
    const plain = target.backend();
    var table = plain.vtable.*;
    table.draft = draft;
    table.probabilities = chances;
    var clock: Clock = .{ .target = &target };
    var engine = Engine.init(a, &cfg, .{ .ptr = plain.ptr, .vtable = &table, .chunk_invariant = plain.chunk_invariant }, clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(a);
    defer log.deinit();
    engine.log = &log;
    var look = try SuffixLookup.init(a, .{ .min_match = 4 }); // the lane host's (core/lane_host.zig min_match)
    defer look.deinit();
    state = .{ .target = &target, .inner = plain.vtable.draft, .prompt_len = scene.prompt.len, .seed = scene.seed, .low_first = scene.low_first, .sure_first = scene.sure_first };
    defer state.asked.deinit(a);
    var s = try sm.Stream.init(a, .{ .id = scene.name, .prompt = scene.prompt, .max_new = scene.max_new, .eos = scene.eos, .proposer = look.proposer() });
    defer s.deinit(a);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();

    var first_depth: ?i64 = null;
    var head_plans: usize = 0;
    var tokens: usize = 1; // the first token, drawn after the prompt pass without a round
    try w.print("{{\"case\":\"{s}\",\"max_new\":{d},\"prompt\":", .{ scene.name, scene.max_new });
    try list(w, scene.prompt);
    try w.writeAll(",\"reply\":");
    try list(w, s.emitted());
    try w.writeAll(",\"asked\":{");
    for (state.asked.keys(), state.asked.values(), 0..) |at, x, i| {
        try w.print("{s}\"{d}\":[", .{ if (i > 0) "," else "", at });
        try list(w, &x.drafts);
        try w.writeAll(",[");
        for (x.chances, 0..) |c, j| try w.print("{s}{d}", .{ if (j > 0) "," else "", c });
        try w.writeAll("]]");
    }
    try w.writeAll("},\"rounds\":[");
    var rounds: usize = 0;
    for (log.events.items) |e| {
        const kind = field(e, "ev");
        if (isStr(kind, "draft") and first_depth == null) first_depth = field(e, "depth").?.int;
        if (isStr(kind, "plan") and isStr(field(e, "kind"), "head")) head_plans += 1;
        if (!isStr(kind, "round")) continue;
        const window = field(e, "window").?.u32s;
        const path = field(e, "path").?.u32s;
        const got = field(e, "got").?.u32s;
        tokens += got.len;
        try w.print("{s}{{\"kind\":\"{s}\",\"rows\":{d},\"kept\":{d},\"tokens\":{d}}}", .{ if (rounds > 0) "," else "", field(e, "kind").?.str, window.len, path.len - 1, got.len });
        rounds += 1;
    }
    const least: u32 = if (engine.alone and cfg.context_copies) 1 else 2;
    try w.print("],\"lanes\":{{\"drafts\":{d},\"first_depth\":{d},\"cut\":{d},\"first_copy\":{d},\"max_copy\":{d},\"enter_match\":{d},\"least_copy\":{d},\"min_match\":{d}}}", .{ cfg.most_drafts, first_depth orelse -1, @as(f64, cut), cfg.first_copy, cfg.max_copy, cfg.enter_match, least, look.params.min_match });
    try w.print(",\"checks\":{{\"hook_calls\":{d},\"head_plans\":{d},\"cut_binds\":{d},\"tokens\":{d},\"emitted\":{d},\"stream_rounds\":{d}}}}}\n", .{ state.hook_calls, head_plans, state.cut_binds, tokens, s.emitted().len, s.rounds });
    return first_depth != null and state.hook_calls == head_plans and tokens == s.emitted().len and rounds == s.rounds;
}

/// `n` tokens of a fixed generator, none of them the fake's close token (91) or the end token (96).
fn random(out: *std.ArrayList(u32), seed: u64, n: usize) !void {
    var x: u64 = seed;
    const end = out.items.len + n;
    while (out.items.len < end) {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        const t: u32 = @intCast((x >> 33) % fake.vocab);
        if (t == 91 or t == 96) continue;
        try out.append(a, t);
    }
}

/// `blocks` of the motif 20..27 each followed by a token of its own, so every copy the motif backs misses at once.
fn motifs(out: *std.ArrayList(u32), blocks: u32) !void {
    for (0..blocks) |k| {
        for (20..28) |t| try out.append(a, @intCast(t));
        try out.append(a, @intCast(30 + k));
    }
}

fn cycle(out: *std.ArrayList(u32), period: []const u32, times: usize) !void {
    for (0..times) |_| try out.appendSlice(a, period);
}

test "open-dneo oracle: a lone stream's rounds under GLM's DFlash2 facts, as JSON lines" {
    const io = std.testing.io;
    std.Io.Dir.cwd().deleteFile(io, out_path) catch {};
    var lists: [13]std.ArrayList(u32) = @splat(.empty);
    defer for (&lists) |*l| l.deinit(a);
    const prompt_a = &lists[0];
    try random(prompt_a, 1, 12);
    const plain = &lists[1]; // no repeats: every round the drafter's
    try random(plain, 2, 200);
    const land = &lists[2]; // four tokens, then a period of five: copies land whole and grow
    try random(land, 3, 4);
    try cycle(land, &.{ 11, 12, 13, 14, 15 }, 40);
    const miss = &lists[3]; // a motif with a new token after it each time: copies miss, four in a row go silent
    try motifs(miss, 20);
    const miss_land = &lists[4]; // misses, then a period of seven
    try motifs(miss_land, 5);
    try cycle(miss_land, &.{ 60, 61, 62, 63, 64, 65, 66 }, 20);
    const prompt_e = &lists[5]; // the reply repeats the prompt's first twenty tokens
    for (70..90) |t| try prompt_e.append(a, @intCast(t));
    try prompt_e.appendSlice(a, &.{ 1, 2, 3 });
    const from_prompt = &lists[6];
    try cycle(from_prompt, prompt_e.items[0..20], 10);
    const ending = &lists[7]; // the end token as the reply's 42nd
    try random(ending, 11, 41);
    try ending.append(a, 96);
    try random(ending, 12, 60);
    const period = &lists[9];
    try random(period, 14, 40);
    const long = &lists[8]; // a period of forty, once with a token changed: copies grow to the widest, one lands in part
    try random(long, 13, 4);
    try cycle(long, period.items, 3);
    for (period.items, 0..) |x, i| try long.append(a, if (i == 30) (if (x + 1 == 91 or x + 1 == 96) (x + 2) % fake.vocab else (x + 1) % fake.vocab) else x);
    try cycle(long, period.items, 4);
    const coin = &lists[10]; // two tokens at random: long matches everywhere, copies that miss, land and go silent
    try random(coin, 15, 4);
    var x: u64 = 16;
    for (0..300) |_| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        try coin.append(a, if (((x >> 40) & 1) == 0) 11 else 12);
    }
    const two = &lists[11]; // copies of two tokens, the least a copy takes
    try random(two, 17, 4);
    try cycle(two, &.{ 11, 12 }, 100);
    const one = &lists[12]; // copies of one token, which none take
    try random(one, 18, 4);
    try cycle(one, &.{55}, 100);

    const scenes = [_]Scene{
        .{ .name = "plain", .prompt = prompt_a.items, .pattern = plain.items, .max_new = 64, .seed = 21 },
        .{ .name = "land", .prompt = prompt_a.items, .pattern = land.items, .max_new = 160, .seed = 22 },
        .{ .name = "miss", .prompt = prompt_a.items, .pattern = miss.items, .max_new = 170, .seed = 23 },
        .{ .name = "miss-land", .prompt = prompt_a.items, .pattern = miss_land.items, .max_new = 175, .seed = 24 },
        .{ .name = "from-prompt", .prompt = prompt_e.items, .pattern = from_prompt.items, .max_new = 120, .seed = 25 },
        .{ .name = "limit-copy", .prompt = prompt_a.items, .pattern = land.items, .max_new = 37, .seed = 26 },
        .{ .name = "limit-head", .prompt = prompt_a.items, .pattern = plain.items, .max_new = 10, .seed = 27 },
        .{ .name = "end-token", .prompt = prompt_a.items, .pattern = ending.items, .max_new = 64, .eos = &.{96}, .seed = 28 },
        .{ .name = "long-period", .prompt = prompt_a.items, .pattern = long.items, .max_new = 300, .seed = 30 },
        .{ .name = "limit-wide", .prompt = prompt_a.items, .pattern = long.items, .max_new = 150, .seed = 31 },
        .{ .name = "coin", .prompt = prompt_a.items, .pattern = coin.items, .max_new = 280, .seed = 32 },
        .{ .name = "period-two", .prompt = prompt_a.items, .pattern = two.items, .max_new = 60, .seed = 33 },
        .{ .name = "period-one", .prompt = prompt_a.items, .pattern = one.items, .max_new = 60, .seed = 34 },
        .{ .name = "sure-first", .prompt = prompt_a.items, .pattern = plain.items, .max_new = 40, .seed = 35, .sure_first = true },
        .{ .name = "sure-first-short", .prompt = prompt_a.items, .pattern = plain.items, .max_new = 3, .seed = 36, .sure_first = true },
        .{ .name = "low-first (seen only)", .prompt = prompt_a.items, .pattern = plain.items, .max_new = 64, .seed = 29, .low_first = true },
    };
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var held: usize = 0;
    for (scenes) |scene| held += @intFromBool(try run(scene, &out.writer));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = out.written() });
    try std.testing.expectEqual(scenes.len, held);
}
