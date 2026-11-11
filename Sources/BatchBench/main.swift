// BatchBench — batch-throughput probe for the RFD4 flow trunk.
//
// Builds the REAL TrunkLayer / PairReprUpdate types from RFD4MLX against a
// synthetic Weights dictionary of the checkpoint's shapes, so what is timed is the
// production forward pass, not a replica of it. No checkpoint needed.
//
//   BB=1,2,4,8  BN=155,638  BREPS=6  ./.build/release/BatchBench
//   BMODE=mem   measures peak memory for one (n, b) and exits
//   BMODE=parts times attention-only and pairupdate-only separately
//
// Measurement rules: everything is interleaved inside one process in shuffled
// order, every timed result is forced with MLX.eval, the first rounds are
// discarded, and the report is a ratio plus the spread.

import Foundation
import MLX
import MLXRandom
import RFD4MLX

// Top-level code in a `main.swift` is @MainActor-isolated under Swift 6, so every
// function here that touches a top-level `let`/`var` has to be isolated too. They
// all do (the synthetic weights, the input cache, the layer stacks), so the
// annotation is blanket rather than case-by-case. The alternative -- building this
// target in Swift 5 language mode -- would leave one target in the package on a
// different language version, which is a worse thing to explain later.

// MARK: - shapes (verified against docs/checkpoint-inventory.md parameter counts)

let TOK = 768       // token dim
let CZ = 384        // pair dim
let CH = 256        // tri-mult hidden
let CCOND = 256     // conditioning dim
let HEADS = 12
let DHEAD = 64
let PT_HIDDEN = 768 // pair transition hidden  (= 2 * c_z; see note in report)
let SWIGLU_IN = 6144

@MainActor
func env(_ k: String) -> String? { ProcessInfo.processInfo.environment[k] }
@MainActor
func ints(_ s: String?) -> [Int] { (s ?? "").split(separator: ",").compactMap { Int($0) } }

let batches = ints(env("BB")).isEmpty ? [1, 2, 4] : ints(env("BB"))
let ns = ints(env("BN")).isEmpty ? [155] : ints(env("BN"))
let reps = Int(env("BREPS") ?? "6")!
let mode = env("BMODE") ?? "trunk"

var rngKey = MLXRandom.key(12345)
@MainActor
func rnd(_ shape: [Int], _ scale: Float = 0.05) -> MLXArray {
    let k = MLXRandom.split(key: rngKey, into: 2)
    rngKey = k[0]
    return MLXRandom.normal(shape, key: k[1]) * scale
}

// MARK: - synthetic checkpoint

@MainActor
func trunkLayerWeights(_ p: String, into d: inout [String: MLXArray]) {
    for blk in ["mhba", "transition"] {
        d["\(p).\(blk).adaln.norm_cond.weight"] = rnd([CCOND], 1.0)
        d["\(p).\(blk).adaln.to_gamma.0.weight"] = rnd([TOK, CCOND])
        d["\(p).\(blk).adaln.to_gamma.0.bias"] = rnd([TOK])
        d["\(p).\(blk).adaln.to_beta.weight"] = rnd([TOK, CCOND])
        d["\(p).\(blk).scale_output.to_adaln_zero_gamma.0.weight"] = rnd([TOK, CCOND])
        d["\(p).\(blk).scale_output.to_adaln_zero_gamma.0.bias"] = rnd([TOK])
    }
    let m = "\(p).mhba.mha"
    d["\(m).node_norm.weight"] = rnd([TOK], 1.0)
    d["\(m).to_qkv.weight"] = rnd([3 * TOK, TOK])
    d["\(m).to_qkv.bias"] = rnd([3 * TOK])
    d["\(m).to_g.weight"] = rnd([TOK, TOK])
    d["\(m).to_g.bias"] = rnd([TOK])
    d["\(m).q_layer_norm.weight"] = rnd([TOK], 1.0)
    d["\(m).k_layer_norm.weight"] = rnd([TOK], 1.0)
    d["\(m).to_out_node.weight"] = rnd([TOK, TOK])
    d["\(m).to_out_node.bias"] = rnd([TOK])
    d["\(m).pair_norm.weight"] = rnd([CZ], 1.0)
    d["\(m).to_bias.weight"] = rnd([HEADS, CZ])
    let t = "\(p).transition.transition"
    d["\(t).swish_linear.0.weight"] = rnd([SWIGLU_IN, TOK])
    d["\(t).linear_out.weight"] = rnd([TOK, SWIGLU_IN / 2])
}

@MainActor
func pairUpdateWeights(_ p: String, into d: inout [String: MLXArray]) {
    d["\(p).layer_norm_in.weight"] = rnd([TOK], 1.0)
    d["\(p).layer_norm_in.bias"] = rnd([TOK])
    d["\(p).linear_x.weight"] = rnd([2 * CZ, TOK])
    for tm in ["tri_mult_out", "tri_mult_in"] {
        let q = "\(p).\(tm)"
        d["\(q).layer_norm_in.weight"] = rnd([CZ], 1.0)
        d["\(q).layer_norm_in.bias"] = rnd([CZ])
        d["\(q).layer_norm_out.weight"] = rnd([CH], 1.0)
        d["\(q).layer_norm_out.bias"] = rnd([CH])
        for l in ["linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g"] {
            d["\(q).\(l).weight"] = rnd([CH, CZ])
        }
        d["\(q).linear_g.weight"] = rnd([CZ, CZ])
        d["\(q).linear_z.weight"] = rnd([CZ, CH])
    }
    let tr = "\(p).transition_out"
    d["\(tr).layer_norm.weight"] = rnd([CZ], 1.0)
    d["\(tr).layer_norm.bias"] = rnd([CZ])
    d["\(tr).linear_1.weight"] = rnd([PT_HIDDEN, CZ])
    d["\(tr).linear_2.weight"] = rnd([CZ, PT_HIDDEN])
}

let LAYERS = 14
let PAIRS = 7

var dict = [String: MLXArray]()
for i in 0 ..< LAYERS { trunkLayerWeights("L\(i)", into: &dict) }
for i in 0 ..< PAIRS { pairUpdateWeights("P\(i)", into: &dict) }
let W = Weights(arrays: dict)

let layers: [TrunkLayer] = try (0 ..< LAYERS).map {
    try TrunkLayer(W.scoped("L\($0)"), heads: HEADS, dimHead: DHEAD, kind: .rms)
}
let pairs: [PairReprUpdate] = try (0 ..< PAIRS).map { try PairReprUpdate(W.scoped("P\($0)")) }
MLX.eval(Array(dict.values))

let weightBytes = dict.values.reduce(0) { $0 + $1.size * 4 }
FileHandle.standardError.write(
    "synthetic trunk: \(dict.count) tensors, \(String(format: "%.2f", Double(weightBytes) / 1e9)) GB\n"
        .data(using: .utf8)!)

// MARK: - the three units of work

struct In {
    let s: MLXArray, z: MLXArray, cond: MLXArray, mask: MLXArray
}

var inputs = [String: In]()
@MainActor
func inputsFor(_ b: Int, _ n: Int) -> In {
    let k = "\(b)x\(n)"
    if let v = inputs[k] { return v }
    let v = In(s: rnd([b, n, TOK], 1.0), z: rnd([b, n, n, CZ], 1.0),
               cond: rnd([b, n, CCOND], 1.0), mask: MLX.ones([b, n]))
    MLX.eval(v.s, v.z, v.cond, v.mask)
    inputs[k] = v
    return v
}

/// One full trunk pass — the loop of `Trunk.applyTrunk`, 14 layers with a pair
/// update after each populated even slot.
@MainActor
func trunkPass(_ i: In) -> [MLXArray] {
    var s = i.s, z = i.z
    var pi = 0
    for l in 0 ..< LAYERS {
        s = layers[l](s, pairRep: z, cond: i.cond, mask: i.mask)
        if l < LAYERS - 1, l % 2 == 0 {
            z = pairs[pi](s, pairRep: z, mask: i.mask)
            pi += 1
        }
    }
    return [s, z]
}

@MainActor
func attentionOnly(_ i: In) -> [MLXArray] {
    var s = i.s
    for l in 0 ..< LAYERS { s = layers[l](s, pairRep: i.z, cond: i.cond, mask: i.mask) }
    return [s]
}

@MainActor
func pairOnly(_ i: In) -> [MLXArray] {
    var z = i.z
    for p in 0 ..< PAIRS { z = pairs[p](i.s, pairRep: z, mask: i.mask) }
    return [z]
}

@MainActor
func work(_ i: In) -> [MLXArray] {
    switch mode {
    case "attn": return attentionOnly(i)
    case "pair": return pairOnly(i)
    default: return trunkPass(i)
    }
}

// MARK: - memory mode

if mode == "mem" {
    let b = batches[0], n = ns[0]
    let i = inputsFor(b, n)
    MLX.eval(trunkPass(i))            // warm
    MLX.GPU.clearCache()
    MLX.GPU.resetPeakMemory()
    let t0 = DispatchTime.now().uptimeNanoseconds
    MLX.eval(trunkPass(i))
    let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
    print(String(format: "MEM b=%d n=%d  peak %.3f GB  active %.3f GB  cache %.3f GB  "
                 + "pass %.3f s  per-design %.3f s",
                 b, n, Double(MLX.GPU.peakMemory) / 1e9,
                 Double(MLX.GPU.activeMemory) / 1e9, Double(MLX.GPU.cacheMemory) / 1e9,
                 dt, dt / Double(b)))
    exit(0)
}

// MARK: - interleaved timing

struct Key: Hashable { let b: Int; let n: Int }
var cases: [Key] = []
for n in ns { for b in batches { cases.append(Key(b: b, n: n)) } }

// Prime every case (Metal kernel compilation + MLX compile tracing), twice.
for _ in 0 ..< 2 { for c in cases { MLX.eval(work(inputsFor(c.b, c.n))) } }

var times = [Key: [Double]]()
var rng = SystemRandomNumberGenerator()
for r in 0 ..< reps {
    // ABBA: forward on even rounds, reversed on odd, then shuffled thereafter.
    let order: [Key] = r == 0 ? cases : (r == 1 ? cases.reversed() : cases.shuffled(using: &rng))
    for c in order {
        let i = inputsFor(c.b, c.n)
        let t0 = DispatchTime.now().uptimeNanoseconds
        MLX.eval(work(i))
        times[c, default: []].append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
    }
    FileHandle.standardError.write("round \(r + 1)/\(reps)\n".data(using: .utf8)!)
}

@MainActor
func med(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
@MainActor
func mn(_ a: [Double]) -> Double { a.min()! }
@MainActor
func mx(_ a: [Double]) -> Double { a.max()! }

@MainActor
func p10(_ a: [Double]) -> Double { let s = a.sorted(); return s[max(0, s.count / 10)] }

print("")
print("== \(mode) pass, fp32, \(reps) rounds, interleaved+shuffled ==")
print("  the machine is shared; MIN is the least-contaminated estimator, so the")
print("  per-design column and the ratio are both built from min, not median.")
print("   n     b     min s    p10 s    med s   per-design(min)  med/min   vs b=1")
for n in ns {
    let base = times[Key(b: batches[0], n: n)].map { mn($0) / Double(batches[0]) }
    for b in batches {
        guard let t = times[Key(b: b, n: n)] else { continue }
        let lo = mn(t), per = lo / Double(b)
        let rel = base.map { per / $0 } ?? 1
        print(String(format: "  %4d  %4d  %8.3f %8.3f %8.3f     %8.4f       %5.2fx  %+6.1f%%",
                     n, b, lo, p10(t), med(t), per, med(t) / lo, 100 * (rel - 1)))
    }
}
print(String(format: "  peak %.2f GB (allocator high-water over the whole run)",
             Double(MLX.GPU.peakMemory) / 1e9))
