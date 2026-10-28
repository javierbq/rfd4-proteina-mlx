import Foundation
import MLX
import MLXFast
import MLXRandom

let n = Int(ProcessInfo.processInfo.environment["BN"] ?? "320")!
let reps = Int(ProcessInfo.processInfo.environment["BREPS"] ?? "10")!
let cz = 384, ch = 256, tok = 768

func rnd(_ s: [Int]) -> MLXArray { MLXRandom.normal(s) * 0.05 }

struct TM {
    let outgoing: Bool
    let niW = rnd([cz]), niB = rnd([cz])
    let noW = rnd([ch]), noB = rnd([ch])
    let zW  = rnd([cz, ch])
    let fW  = rnd([4 * ch + cz, cz])      // a_p a_g b_p b_g g
    // channels-first split: ab-part and g-part
    var abW: MLXArray { fW[0 ..< (4 * ch)] }
    var gW:  MLXArray { fW[(4 * ch) ..< (4 * ch + cz)] }
}
let tmOut = TM(outgoing: true), tmIn = TM(outgoing: false)
let trNW = rnd([cz]), trNB = rnd([cz])
let w1 = rnd([2 * cz, cz]), b1 = rnd([2 * cz])
let w2 = rnd([cz, 2 * cz]), b2 = rnd([cz])
let puNW = rnd([tok]), puNB = rnd([tok]), puXW = rnd([2 * cz, tok])
let zIn = rnd([1, n, n, cz]), xIn = rnd([1, n, tok])
let mask = MLX.ones([1, n])
MLX.eval(zIn, xIn, mask, tmOut.fW, tmOut.niW, tmOut.niB, tmOut.noW, tmOut.noB, tmOut.zW,
         tmIn.fW, tmIn.niW, tmIn.niB, tmIn.noW, tmIn.noB, tmIn.zW,
         trNW, trNB, w1, b1, w2, b2, puNW, puNB, puXW,
         tmOut.abW, tmOut.gW, tmIn.abW, tmIn.gW)

@MainActor func lin(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
    let y = MLX.matmul(x, w.transposed()); return b.map { y + $0 } ?? y
}
@MainActor func linAdd(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    MLX.addMM(b, x, w.transposed())
}
@MainActor func lnFast(_ x: MLXArray, _ w: MLXArray?, _ b: MLXArray?) -> MLXArray {
    MLXFast.layerNorm(x, weight: w, bias: b, eps: 1e-5)
}
@MainActor func lnHand(_ x: MLXArray, _ w: MLXArray?, _ b: MLXArray?) -> MLXArray {
    let mu = MLX.mean(x, axis: -1, keepDims: true)
    let c = x - mu
    let v = MLX.mean(c * c, axis: -1, keepDims: true)
    var y = c * MLX.rsqrt(v + 1e-5); if let w { y = y * w }; if let b { y = y + b }
    return y
}

struct Opt {
    var handLN = false          // pre-MLXFast layerNorm
    var dropFinalMask = false   // (z + transition(z)) * pm  ->  z + transition(z)
    var noMask = false          // all-ones mask: skip every mask multiply
    var cfProj = false          // channels-first projection, no transposeIn copies
    var addmmBias = false       // fuse PairTransition biases into the GEMM epilogue
}

@MainActor func ln(_ o: Opt, _ x: MLXArray, _ w: MLXArray?, _ b: MLXArray?) -> MLXArray {
    o.handLN ? lnHand(x, w, b) : lnFast(x, w, b)
}

// ---- triangle multiplication
@MainActor func triMul(_ o: Opt, _ t: TM, _ z: MLXArray, _ pairMask: MLXArray) -> MLXArray {
    let m = pairMask.expandedDimensions(axis: -1)
    let zn = ln(o, z, t.niW, t.niB)

    if o.cfProj {
        // [4c, n*n] contiguous: W @ znT, znT is a no-copy col-major view
        let znT = zn.reshaped(n * n, cz).transposed()
        let abCF = MLX.matmul(t.abW, znT).reshaped(4 * ch, n, n)
        let gl   = lin(zn, t.gW)                        // natural layout for the gate
        let ap = abCF[0 ..< ch], ag = abCF[ch ..< 2*ch]
        let bp = abCF[2*ch ..< 3*ch], bg = abCF[3*ch ..< 4*ch]
        let mc = o.noMask ? nil : pairMask              // [1,n,n] broadcasts over [c,n,n]
        var a = ap * MLX.sigmoid(ag); if let mc { a = a * mc }
        var b = bp * MLX.sigmoid(bg); if let mc { b = b * mc }
        // outgoing: a[c,i,k] b[c,j,k] -> a @ bᵀ ; incoming: a[c,k,i] b[c,k,j] -> aᵀ @ b
        let prod = t.outgoing ? MLX.matmul(a, b.transposed(0, 2, 1))
                              : MLX.matmul(a.transposed(0, 2, 1), b)
        var x = prod.transposed(1, 2, 0).expandedDimensions(axis: 0)   // [1,i,j,c]
        x = ln(o, x, t.noW, t.noB)
        x = lin(x, t.zW)
        return x * MLX.sigmoid(gl)
    }

    let fused = lin(zn, t.fW)
    let ap = fused[.ellipsis, 0 ..< ch], ag = fused[.ellipsis, ch ..< 2*ch]
    let bp = fused[.ellipsis, 2*ch ..< 3*ch], bg = fused[.ellipsis, 3*ch ..< 4*ch]
    let gl = fused[.ellipsis, 4*ch ..< (4*ch + cz)]
    var a = ap * MLX.sigmoid(ag), b = bp * MLX.sigmoid(bg)
    if !o.noMask { a = a * m; b = b * m }
    let (am, bm) = t.outgoing ? (a.transposed(0, 3, 1, 2), b.transposed(0, 3, 2, 1))
                              : (a.transposed(0, 3, 2, 1), b.transposed(0, 3, 1, 2))
    var x = MLX.matmul(am, bm).transposed(0, 2, 3, 1)
    x = ln(o, x, t.noW, t.noB)
    x = lin(x, t.zW)
    return x * MLX.sigmoid(gl)
}

@MainActor func transition(_ o: Opt, _ z: MLXArray, _ pairMask: MLXArray) -> MLXArray {
    let m = pairMask.expandedDimensions(axis: -1)
    var y = ln(o, z, trNW, trNB)
    y = MLX.maximum(o.addmmBias ? linAdd(y, w1, b1) : lin(y, w1, b1), 0)
    let out = o.addmmBias ? linAdd(y, w2, b2) : lin(y, w2, b2)
    return o.noMask ? out : out * m
}

@MainActor func pairUpdate(_ o: Opt, _ x: MLXArray, _ pairRep: MLXArray, _ mask: MLXArray) -> MLXArray {
    let pairMask = mask.expandedDimensions(axis: 1) * mask.expandedDimensions(axis: 2)
    let pm = pairMask.expandedDimensions(axis: -1)
    let xm = x * mask.expandedDimensions(axis: -1)
    let proj = lin(ln(o, xm, puNW, puNB), puXW)
    let pi = proj[.ellipsis, 0 ..< cz], pj = proj[.ellipsis, cz ..< 2*cz]
    var z = pairRep + pi.expandedDimensions(axis: 1) * pj.expandedDimensions(axis: 2)
    if !o.noMask { z = z * pm }
    z = z + triMul(o, tmOut, z, pairMask); if !o.noMask { z = z * pm }
    z = z + triMul(o, tmIn,  z, pairMask); if !o.noMask { z = z * pm }
    z = z + transition(o, z, pairMask)
    if !o.noMask && !o.dropFinalMask { z = z * pm }
    return z
}

@MainActor func bench(_ name: String, _ o: Opt) -> Double {
    for _ in 0 ..< 2 { MLX.eval(pairUpdate(o, xIn, zIn, mask)) }
    var best = Double.infinity
    for _ in 0 ..< reps {
        let t0 = DispatchTime.now().uptimeNanoseconds
        MLX.eval(pairUpdate(o, xIn, zIn, mask))
        best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
    }
    print(String(format: "  %-46s %8.1f ms", (name as NSString).utf8String!, best * 1000))
    fflush(stdout)
    return best
}

// numeric check: every exact variant must match the baseline
@MainActor func check(_ name: String, _ o: Opt) {
    let r0 = pairUpdate(Opt(), xIn, zIn, mask)
    let r1 = pairUpdate(o, xIn, zIn, mask)
    let d = MLX.max(MLX.abs(r0 - r1)).item(Float.self)
    let s = MLX.max(MLX.abs(r0)).item(Float.self)
    print(String(format: "  %-46s max|Δ| %.3e   (max|z| %.3e)",
                 (name as NSString).utf8String!, d, s))
}

let order = ["base", "base2", "handln", "nofinal", "nomask", "cfproj", "addmm", "nomask_addmm", "cf_addmm", "all"]
let opts: [String: Opt] = [
    "base":    Opt(),
    "base2":   Opt(),                       // duplicate control -> noise floor
    "handln":  Opt(handLN: true),
    "nofinal": Opt(dropFinalMask: true),
    "nomask":  Opt(noMask: true),
    "cfproj":  Opt(cfProj: true),
    "addmm":   Opt(addmmBias: true),
    "all":     Opt(dropFinalMask: true, noMask: true, cfProj: true, addmmBias: true),
    "nomask_addmm": Opt(dropFinalMask: true, noMask: true, addmmBias: true),
    "cf_addmm":     Opt(cfProj: true, addmmBias: true),
]
if ProcessInfo.processInfo.environment["BVAR"] == "check" {
    for k in order.dropFirst(2) {
        let r0 = pairUpdate(Opt(), xIn, zIn, mask)
        let r1 = pairUpdate(opts[k]!, xIn, zIn, mask)
        let d = MLX.max(MLX.abs(r0 - r1)).item(Float.self)
        let m = MLX.max(MLX.abs(r0)).item(Float.self)
        print(String(format: "  %-16s max|delta| %.3e   (max|z| %.3e)",
                     (k as NSString).utf8String!, d, m))
    }
    exit(0)
}
for _ in 0 ..< 3 { for k in order { MLX.eval(pairUpdate(opts[k]!, xIn, zIn, mask)) } }
var times: [String: [Double]] = [:]
var rng = SystemRandomNumberGenerator()
for r in 0 ..< reps {
    for k in order.shuffled(using: &rng) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        MLX.eval(pairUpdate(opts[k]!, xIn, zIn, mask))
        times[k, default: []].append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
    }
    FileHandle.standardError.write("round \(r + 1)/\(reps)\n".data(using: .utf8)!)
}
func med(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
func p25(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 4] }
let baseM = med(times["base"]!), baseQ = p25(times["base"]!)
print("")
print(String(format: "== PairReprUpdate, n=%d, fp32, %d rounds, shuffled order ==", n, reps))
print("  variant          p25 ms    med ms   d(p25)    d(med)")
for k in order {
    let m = med(times[k]!), q = p25(times[k]!)
    print(String(format: "  %-14s %8.1f  %8.1f  %+7.1f%%  %+7.1f%%",
                 (k as NSString).utf8String!, q * 1000, m * 1000,
                 100 * (q - baseQ) / baseQ, 100 * (m - baseM) / baseM))
}
print(String(format: "  peak %.2f GB", Double(MLX.GPU.peakMemory) / 1e9))
