import Foundation

/// Writes an `AtomArray` as mmCIF.
///
/// The counterpart of `MMCIF.parse`, and deliberately minimal: enough `atom_site`
/// columns that a viewer or a downstream tool reads the structure correctly, with
/// no attempt at a complete deposition-grade file.
///
/// Coordinates are written in **Angstrom**. The model works in nanometres
/// throughout, so the conversion happens here, once, at the boundary — rather than
/// being scattered through the pipeline where a missed factor of ten produces a
/// structure that looks plausible and is wrong by an order of magnitude.
public enum CIFWriter {
    public static let nanometresToAngstrom: Float = 10

    /// - Parameters:
    ///   - array: atoms to write; coordinates in nanometres.
    ///   - name: the data block name.
    ///   - bFactors: optional per-atom value, written to `B_iso_or_equiv`.
    public static func string(
        from array: AtomArray, name: String = "generated", bFactors: [Float]? = nil
    ) -> String {
        var out = """
        data_\(name)
        #
        loop_
        _atom_site.group_PDB
        _atom_site.id
        _atom_site.type_symbol
        _atom_site.label_atom_id
        _atom_site.label_alt_id
        _atom_site.label_comp_id
        _atom_site.label_asym_id
        _atom_site.label_entity_id
        _atom_site.label_seq_id
        _atom_site.pdbx_PDB_ins_code
        _atom_site.Cartn_x
        _atom_site.Cartn_y
        _atom_site.Cartn_z
        _atom_site.occupancy
        _atom_site.B_iso_or_equiv
        _atom_site.pdbx_PDB_model_num

        """

        let chain = array.chainID ?? [String](repeating: "A", count: array.count)
        let resID = array.resID ?? [Int](repeating: 1, count: array.count)

        for i in array.indices {
            let p = array.coord[i]
            let element = array.element?[i]
                ?? String(array.atomName[i].prefix(1))
            // An atom name containing a quote must be quoted, or the file will not
            // survive its own reader. Nucleotide names (C1', O5') are the usual case.
            let atom = array.atomName[i].contains("'")
                ? "\"\(array.atomName[i])\"" : array.atomName[i]
            let b = bFactors.map { $0[i] } ?? 0

            out += String(
                format: "ATOM %d %@ %@ . %@ %@ 1 %d ? %.3f %.3f %.3f 1.00 %.2f 1\n",
                i + 1, element, atom, array.resName[i], chain[i], resID[i],
                p.x * nanometresToAngstrom, p.y * nanometresToAngstrom,
                p.z * nanometresToAngstrom, b)
        }
        out += "#\n"
        return out
    }

    public static func write(
        _ array: AtomArray, to url: URL, name: String = "generated",
        bFactors: [Float]? = nil
    ) throws {
        try string(from: array, name: name, bFactors: bFactors)
            .write(to: url, atomically: true, encoding: .utf8)
    }
}
