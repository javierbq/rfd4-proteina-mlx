import Foundation

/// The two parse-time transforms the design path depends on.
///
/// Upstream applies these in the parser, as flags on the `ParseConfig` that
/// `load_condition_spec_inputs` uses — `RCSB_WITH_METALS`, whose defaults include
/// `convert_mse_to_met=True` and `altloc="random_clash_aware"`. Checked rather than
/// assumed: that loader takes `as_annotated_cif=False` by default, so it is the
/// RCSB preset and not `ANNOTATED_CIF` (which turns both off) that governs a design
/// input.
///
/// They are separate methods here rather than folded into `MMCIF.fromMMCIF` so the
/// reader stays a faithful transcription of the file — its parity tests compare it
/// column for column against biotite — and the design path states which
/// normalisations it is applying.
extension AtomArray {

    /// `convert_mse_to_met`: selenomethionine read as methionine.
    ///
    /// MSE is how a crystallographer solves the phase problem, so it appears in a
    /// large fraction of RCSB entries — and it is not in the canonical-20 alphabet.
    /// Without this the residue's atoms all fail the vocabulary lookup, the residue
    /// vanishes from the chain, and its neighbours become adjacent: a shorter layout,
    /// and a peptide bond asserted across a gap the structure does not have.
    ///
    /// Upstream renames the residue, clears the hetero flag, renames `SE` to `SD`
    /// and sets that atom's element to `S` (`atomworks/io/transforms/atom_array.py:
    /// mse_to_met`). It also reorders the residue's atoms canonically, which does not
    /// matter here because atom37 slots are looked up by name.
    public func convertingMSEToMET() -> AtomArray {
        guard resName.contains("MSE") else { return self }
        var out = self
        for i in indices where resName[i] == "MSE" {
            out.resName[i] = "MET"
            if atomName[i].trimmingCharacters(in: .whitespaces) == "SE" {
                out.atomName[i] = "SD"
                if out.element != nil { out.element![i] = "S" }
            }
        }
        return out
    }

    /// `altloc`: one conformer per atom.
    ///
    /// A residue modelled in two conformations contributes two rows for the same
    /// atom. Keeping both makes an atom37 slot hold whichever came last, inflates a
    /// coordinate group past its distinct-atom count, and double-counts a hotspot —
    /// which trips the CLI's exact-count guard and rejects a perfectly valid target.
    ///
    /// Upstream's rule is `random_clash_aware` with `altloc_seed=0`, which resolves
    /// a whole residue's conformers together and considers clashes. This keeps the
    /// first conformer per atom, preferring an unlabelled row. **Measured** against
    /// upstream on the shipped target with a two-conformer TYR56 OH: it collapsed to
    /// one atom and kept conformer `A` even though `B` carried the higher occupancy
    /// (0.60 against 0.40), which is what this reproduces. The two rules can still
    /// disagree where conformers clash, and that is a known divergence rather than a
    /// claim of parity.
    public func collapsingAltlocs() -> AtomArray {
        guard let alt = altLocID else { return self }
        let blanks: Set<String> = [".", "?", "", " "]
        guard alt.contains(where: { !blanks.contains($0) }) else { return self }

        struct Key: Hashable { let chain: String; let residue: Int; let atom: String }
        var chosen: [Key: (index: Int, altloc: String)] = [:]
        for i in indices {
            let key = Key(chain: chainID?[i] ?? "",
                          residue: resID?[i] ?? -1,
                          atom: atomName[i].trimmingCharacters(in: .whitespaces))
            let id = alt[i]
            guard let held = chosen[key] else { chosen[key] = (i, id); continue }
            // An unlabelled row wins outright; otherwise the lexicographically first
            // id wins, so the choice does not depend on file order.
            let heldIsBlank = blanks.contains(held.altloc)
            let thisIsBlank = blanks.contains(id)
            if thisIsBlank && !heldIsBlank { chosen[key] = (i, id) }
            else if thisIsBlank == heldIsBlank && id < held.altloc { chosen[key] = (i, id) }
        }
        let keep = Set(chosen.values.map(\.index))
        guard keep.count < count else { return self }
        return filtered(indices.filter { keep.contains($0) })
    }

    /// A new array holding only `kept`, in the order given.
    public func filtered(_ kept: [Int]) -> AtomArray {
        var out = AtomArray(resName: kept.map { resName[$0] },
                            atomName: kept.map { atomName[$0] },
                            coord: kept.map { coord[$0] })
        out.element = element.map { c in kept.map { c[$0] } }
        out.tokIdx = tokIdx.map { c in kept.map { c[$0] } }
        out.chainID = chainID.map { c in kept.map { c[$0] } }
        out.resID = resID.map { c in kept.map { c[$0] } }
        out.occupancy = occupancy.map { c in kept.map { c[$0] } }
        out.bFactor = bFactor.map { c in kept.map { c[$0] } }
        out.chainType = chainType.map { c in kept.map { c[$0] } }
        out.atomize = atomize.map { c in kept.map { c[$0] } }
        out.tokenType = tokenType.map { c in kept.map { c[$0] } }
        out.altLocID = altLocID.map { c in kept.map { c[$0] } }
        return out
    }

    /// Both transforms, as the design path needs them.
    public func normalizedForDesign() -> AtomArray {
        collapsingAltlocs().convertingMSEToMET()
    }
}
