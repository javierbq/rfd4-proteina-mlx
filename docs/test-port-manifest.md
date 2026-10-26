# RFD4-Proteina test suite -> Swift/MLX port manifest

Generated from `/private/tmp/claude-501/-Users-javier/b4675505-3fa2-5cb2-b5a1-0f318ce3a657/scratchpad/rfd4-src/RFD4-Proteina`. Sources read as text; nothing imported.

**348 test functions across 67 files.**

| verdict | files | test fns |
|---|---|---|
| PORT | 44 | 233 |
| REVIEW | 0 | 0 |
| DROP | 23 | 115 |

## Port these

| file | LOC | tests | deps | note |
|---|---|---|---|---|
| `e2e/test_bioir_equivalence.py` | 274 | 4 | atomworks, biotite, cuda, torch |  |
| `e2e/test_length_bucketing_e2e.py` | 89 | 4 | atomworks, torch |  |
| `inference_utils/test_condition_resolution.py` | 74 | 4 | atomworks |  |
| `inference_utils/test_condition_spec.py` | 222 | 10 | atomworks |  |
| `inference_utils/test_contig_spec.py` | 174 | 9 | atomworks |  |
| `inference_utils/test_skip_existing.py` | 95 | 4 | atomworks, torch |  |
| `nn/test_pairbias_variants.py` | 118 | 2 | cuda, torch |  |
| `pipeline/test_bond_path_lengths.py` | 314 | 6 | atomworks, biotite |  |
| `pipeline/test_canonical_polymer_ordering.py` | 129 | 3 | atomworks, biotite |  |
| `pipeline/test_expandable_segment_pipeline.py` | 768 | 3 | atomworks, cuda, torch |  |
| `pipeline/test_glycan_loader.py` | 40 | 1 | biotite |  |
| `pipeline/test_nan_token_bond_holes.py` | 76 | 1 | atomworks |  |
| `pipeline/test_pipeline_invariants.py` | 388 | 5 | atomworks, biotite |  |
| `pipeline/test_validation_pipeline_serialization.py` | 14 | 1 | - |  |
| `test_atom37_encoding_alignment.py` | 217 | 5 | atomworks, biotite |  |
| `test_bioir_decoder_policy.py` | 139 | 4 | atomworks, torch |  |
| `test_bioir_lora_merge.py` | 30 | 1 | torch |  |
| `test_inference_integration.py` | 897 | 18 | atomworks, biotite, torch |  |
| `test_inference_nn_overrides.py` | 127 | 9 | checkpoint, torch |  |
| `test_inverse_folding_sampling.py` | 171 | 5 | torch |  |
| `test_model_masking.py` | 859 | 20 | atomworks, torch |  |
| `test_nucleic_ss_input.py` | 33 | 1 | - |  |
| `test_partial_schedule.py` | 170 | 7 | torch |  |
| `test_quickstart_preset.py` | 46 | 3 | - |  |
| `test_recycle_depths.py` | 54 | 6 | - |  |
| `test_skills.py` | 55 | 4 | - |  |
| `transforms/test_budget.py` | 53 | 1 | atomworks |  |
| `transforms/test_conditions_and_modifiers.py` | 659 | 20 | atomworks, biotite |  |
| `transforms/test_design_task_speed.py` | 83 | 1 | - |  |
| `transforms/test_design_tasks.py` | 407 | 12 | atomworks |  |
| `transforms/test_double_bond_stereochemistry.py` | 75 | 3 | atomworks, biotite |  |
| `transforms/test_metals_and_aromatic_dropout.py` | 113 | 5 | - |  |
| `transforms/test_nucleic_hbonds.py` | 51 | 1 | atomworks, biotite |  |
| `transforms/test_pi_ring_conditioning.py` | 118 | 2 | atomworks |  |
| `transforms/test_protein_pocket_sampling.py` | 47 | 1 | biotite |  |
| `transforms/test_secondary_structure_condition.py` | 92 | 3 | atomworks |  |
| `transforms/test_unindexing.py` | 371 | 6 | atomworks, biotite |  |
| `transforms/test_unindexing_atom_matching.py` | 87 | 1 | biotite |  |
| `utils/test_atom_permutation.py` | 415 | 11 | atomworks, biotite, torch |  |
| `utils/test_coordinate_transfer.py` | 251 | 1 | atomworks, biotite |  |
| `utils/test_structure_files.py` | 47 | 4 | - |  |
| `utils/test_subunit_permutation.py` | 193 | 11 | atomworks, biotite |  |
| `utils/test_torch_utils.py` | 139 | 4 | torch |  |
| `utils/test_weight_loading.py` | 202 | 6 | torch |  |

## Need a decision

| file | LOC | tests | deps | note |
|---|---|---|---|---|

## Not portable

| file | LOC | tests | deps | note |
|---|---|---|---|---|
| `callbacks/test_ema.py` | 167 | 6 | torch | Lightning training callbacks |
| `callbacks/test_ema_accumulation.py` | 37 | 1 | torch | Lightning training callbacks |
| `e2e/test_af2_acceptance.py` | 549 | 2 | atomworks, biotite, cuda | needs AlphaFold2 - no AF2 port exists |
| `e2e/test_af3_structure_quality.py` | 534 | 4 | atomworks, biotite, cuda | needs AF3/OpenFold3 baselines |
| `e2e/test_entrypoint_imports_resolve.py` | 49 | 1 | - | Hydra config resolution - no Hydra in Swift |
| `e2e/test_entrypoints_resolve_configs.py` | 42 | 1 | - | Hydra config resolution - no Hydra in Swift |
| `e2e/test_gpu_regression.py` | 282 | 6 | cuda | CUDA regression baselines |
| `e2e/test_gradient_coverage.py` | 143 | 1 | atomworks, cuda | backward pass - inference-only port has none |
| `e2e/test_slurm_launchers.py` | 114 | 4 | - | SLURM job submission - no meaning on device |
| `metrics/test_cath_metric.py` | 131 | 3 | atomworks, biotite | eval suite, needs oracle environments |
| `metrics/test_cdr_torsions_data.py` | 57 | 4 | - | eval suite, needs oracle environments |
| `metrics/test_ddg_salt_metrics.py` | 85 | 1 | atomworks | eval suite, needs oracle environments |
| `metrics/test_developability_metrics.py` | 162 | 10 | atomworks, biotite | eval suite, needs oracle environments |
| `metrics/test_internal_coords.py` | 347 | 8 | atomworks, biotite, torch | eval suite, needs oracle environments |
| `metrics/test_posebusters_metrics.py` | 55 | 1 | atomworks | eval suite, needs oracle environments |
| `metrics/test_soft_threshold.py` | 30 | 2 | - | eval suite, needs oracle environments |
| `metrics/test_sse_metrics.py` | 168 | 3 | atomworks | eval suite, needs oracle environments |
| `metrics/test_tmol_ddg_metrics.py` | 169 | 10 | atomworks, biotite, cuda, torch | eval suite, needs oracle environments |
| `nn/test_triangle_cueq.py` | 94 | 3 | cuda, torch | NVIDIA fused kernel vs fallback - fallback IS the reference |
| `test_oracles.py` | 528 | 10 | atomworks, biotite, cuda | needs AF2/AF3/OpenFold3 as an external oracle |
| `test_setup_search_envs.py` | 125 | 7 | cuda | conda env provisioning |
| `test_sweep_and_scripts.py` | 296 | 22 | - | repo tooling / shell scripts |
| `utils/test_torch_compile.py` | 93 | 5 | cuda, torch | torch.compile plumbing - MLX lazy graph replaces it |

## Full test inventory


### `e2e/test_bioir_equivalence.py` — PORT (4 tests)

- [ ] `test_bioir_designs_are_sound`
- [ ] `test_bioir_first_step_velocity_matches_native`
- [ ] `test_bioir_graph_replay_follows_every_batch`
- [ ] `test_bioir_search_is_sound`

### `e2e/test_length_bucketing_e2e.py` — PORT (4 tests)

- [ ] `test_length_bucketing_requires_ordered_loading`
- [ ] `test_length_bucketing_end_to_end`
- [ ] `test_count_lengths`
- [ ] `test_invalid_count_lengths`

### `inference_utils/test_condition_resolution.py` — PORT (4 tests)

- [ ] `test_alias_and_systematic_name_resolve_to_the_same_class`
- [ ] `test_enum_value_coercion_accepts_equivalent_forms`
- [ ] `test_presence_authored_conditions_write_unified_annotations`
- [ ] `test_presence_authoring_only_updates_selected_coordinate_values`

### `inference_utils/test_condition_spec.py` — PORT (10 tests)

- [ ] `test_end_to_end_condition_spec_survives_cif_round_trip`
- [ ] `test_group_field_merges_disjoint_selections_and_assigns_fresh_ids`
- [ ] `test_grouped_and_plain_condition_entries_are_last_write_wins`
- [ ] `test_condition_application_errors`
- [ ] `test_quoted_boolean_condition_values_keep_their_literal_meaning`
- [ ] `test_mask_only_sequence_conditioning_reveals_the_real_residue_names`
- [ ] `test_unrecognized_example_field_raises`
- [ ] `test_split_global_args_rejects_non_mapping_entries`
- [ ] `test_chain_id_collision_across_input_files_raises`
- [ ] `test_multi_input_files_get_independent_molecule_and_index_groups`

### `inference_utils/test_contig_spec.py` — PORT (9 tests)

- [ ] `test_chain_run_first_assignment_not_left_to_right`
- [ ] `test_original_res_id_snapshot_survives_expandable_segment_shift`
- [ ] `test_wholly_expandable_run_gets_fresh_chain_id_and_grows_contiguously`
- [ ] `test_wholly_expandable_leading_runs_deferred_until_a_later_run_has_preserved_atoms`
- [ ] `test_unsuffixed_expandable_segment_defaults_to_anchor_chain_type_not_polypeptide`
- [ ] `test_anchorless_run_takes_its_expandable_segments_suffix`
- [ ] `test_anchor_chain_with_mixed_chain_types_raises`
- [ ] `test_contig_with_no_preserved_segments_raises`
- [ ] `test_invalid_contigs_raise_clear_errors`

### `inference_utils/test_skip_existing.py` — PORT (4 tests)

- [ ] `test_partition_existing_outputs_routes_existing_designs_to_evaluation`
- [ ] `test_resumed_chain_export_preserves_encoding`
- [ ] `test_partition_existing_outputs_regenerates_empty_cifs`
- [ ] `test_concatenate_results_merges_archived_and_newest_metrics`

### `nn/test_pairbias_variants.py` — PORT (2 tests)

- [ ] `test_sdpa_matches_einsum_reference`
- [ ] `test_cuequivariance_matches_reference`

### `pipeline/test_bond_path_lengths.py` — PORT (6 tests)

- [ ] `test_indexed_res_id_gap_reads_as_the_gap_size`
- [ ] `test_unindexed_motif_spacing_matches_training`
- [ ] `test_no_path_and_too_far_are_told_apart`
- [ ] `test_unindexing_preserves_bonds_within_a_motif`
- [ ] `test_excision_bonds_an_indexed_flank_to_a_designed_one`
- [ ] `test_excision_only_closes_the_gap_a_designed_flank_can_absorb`

### `pipeline/test_canonical_polymer_ordering.py` — PORT (3 tests)

- [ ] `test_closure_annotations`
- [ ] `test_annotation_consistency`
- [ ] `test_masked_ring_closure_runs_high_to_low`

### `pipeline/test_expandable_segment_pipeline.py` — PORT (3 tests)

- [ ] `test_train_val_pipeline_feats_match`
- [ ] `test_train_save_load_val_pipeline_feats_match`
- [ ] `test_expandable_segment_features_match_unconditional`

### `pipeline/test_glycan_loader.py` — PORT (1 tests)

- [ ] `test_glycan_selection`

### `pipeline/test_nan_token_bond_holes.py` — PORT (1 tests)

- [ ] `test_flanking_residues_of_a_dropped_nan_token_stay_two_hops_apart`

### `pipeline/test_pipeline_invariants.py` — PORT (5 tests)

- [ ] `test_pooled_masks_distinguish_atomized_and_in_context_tokens`
- [ ] `test_pipeline_invariants`
- [ ] `test_atomized_residue_is_not_a_res_id_gap`
- [ ] `test_bond_length_invariant_rejects_long_bond`
- [ ] `test_bond_length_invariant_allows_reasonable_or_unmeasured_bonds`

### `pipeline/test_validation_pipeline_serialization.py` — PORT (1 tests)

- [ ] `test_validation_pipeline_serialization`

### `test_atom37_encoding_alignment.py` — PORT (5 tests)

- [ ] `test_vocabulary_slots_ignore_chain_classification`
- [ ] `test_unified_atom37_slot_1_is_center_atom`
- [ ] `test_roundtrip_protein_structure`
- [ ] `test_roundtrip_nucleic_acid_structure`
- [ ] `test_roundtrip_mixed_protein_na_structure`

### `test_bioir_decoder_policy.py` — PORT (4 tests)

- [ ] `test_policy_matches_native_ae_without_reference_coordinates`
- [ ] `test_sliced_presence_preserves_original_rows`
- [ ] `test_formatter_preserves_predicted_coordinates_ligand_and_insertion_head`
- [ ] `test_presence_hook_is_removed_on_error`

### `test_bioir_lora_merge.py` — PORT (1 tests)

- [ ] `test_lora_adapters_are_merged_into_the_base_weights_for_bioir`

### `test_inference_integration.py` — PORT (18 tests)

- [ ] `test_beam_save_resume_defaults`
- [ ] `test_beam_save_resume_requires_search`
- [ ] `test_beam_preset_accepts_save_resume_overrides`
- [ ] `test_design_problem_dataset_basic`
- [ ] `test_duplicate_cif_stems_are_qualified_by_parent_directory`
- [ ] `test_compressed_cif_ids_drop_all_extensions`
- [ ] `test_molecule_entities_from_annotated_cif`
- [ ] `test_example_id_scheme`
- [ ] `test_subset_by_condition_spec_keys`
- [ ] `test_save_resume_and_save`
- [ ] `test_generation_with_replicates_and_prefix_and_refolding`
- [ ] `test_generation_with_subsetting`
- [ ] `test_skip_existing_prevents_regeneration`
- [ ] `test_seeded_generation_is_deterministic`
- [ ] `test_output_cif_annotations`
- [ ] `test_partial_diffusion`
- [ ] `test_motif_conditioning_persists_through_partial_diffusion`
- [ ] `test_inverse_folding`

### `test_inference_nn_overrides.py` — PORT (9 tests)

- [ ] `test_null_overrides_leave_the_checkpoint_unread`
- [ ] `test_overrides_merge_into_the_saved_nn_config`
- [ ] `test_unknown_override_is_rejected`
- [ ] `test_inference_config_exposes_exactly_the_overridable_keys`
- [ ] `test_loader_passes_the_merged_config_to_lightning`
- [ ] `test_hub_bundle_receives_the_overrides`
- [ ] `test_from_pretrained_merges_overrides_into_the_bundle_nn_config`
- [ ] `test_unknown_compile_mode_fails_before_anything_loads`
- [ ] `test_overrides_cannot_be_applied_to_a_supplied_model`

### `test_inverse_folding_sampling.py` — PORT (5 tests)

- [ ] `test_forward_marginal_uses_centering_once_with_groups`
- [ ] `test_inverse_folding_resolution_is_immutable`
- [ ] `test_inverse_folding_rejects_partial_diffusion_overrides`
- [ ] `test_inverse_folding_rejects_unknown_method`
- [ ] `test_partial_simulation_uses_resampled_backbone`

### `test_model_masking.py` — PORT (20 tests)

- [ ] `test_current_helpers_match_every_grid_cell`
- [ ] `test_decoder_rejects_mismatched_residue_encoding`
- [ ] `test_autoencoder_rejects_mismatched_residue_encoding`
- [ ] `test_flow_passes_full_axis_through_autoencoder`
- [ ] `test_autoencoder_context_influence_matches_grid`
- [ ] `test_decoder_coordinates_are_zero_outside_emit_mask`
- [ ] `test_flow_attention_includes_every_nonpadding_context`
- [ ] `test_flow_corruption_uses_flow_emit_masks`
- [ ] `test_full_simulation_preserves_flow_emit_masks_through_trajectory`
- [ ] `test_partial_simulation_passes_flow_emit_masks_at_each_step`
- [ ] `test_beam_initialization_uses_masks_and_centering_groups`
- [ ] `test_beam_rejects_invalid_saved_state_before_reseeding`
- [ ] `test_beam_rejects_invalid_saved_tensors_before_reseeding`
- [ ] `test_flow_predictions_and_simulation_vectors_use_flow_emit_masks`
- [ ] `test_guidance_delta_uses_flow_emit_mask`
- [ ] `test_flow_loss_changes_only_for_supervised_grid_cells`
- [ ] `test_sequence_prediction_loss_uses_local_latent_supervision_mask`
- [ ] `test_autoencoder_losses_change_only_for_supervised_grid_cells`
- [ ] `test_final_formatting_matches_flow_emit_contract`
- [ ] `test_sequence_prediction_conditions_decoder_through_batch`

### `test_nucleic_ss_input.py` — PORT (1 tests)

- [ ] `test_parse_dot_bracket`

### `test_partial_schedule.py` — PORT (7 tests)

- [ ] `test_schedule_starts_at_the_noised_time`
- [ ] `test_fraction_corruption_uses_configured_grid`
- [ ] `test_fraction_preserves_full_schedule_tail`
- [ ] `test_fraction_rejects_zero_remaining_steps`
- [ ] `test_fraction_rejects_mismatched_corruption_time`
- [ ] `test_fraction_rejects_invalid_step_count`
- [ ] `test_fractional_generation_corrupts_at_first_integration_time`

### `test_quickstart_preset.py` — PORT (3 tests)

- [ ] `test_quickstart_matches_the_manual_incantation`
- [ ] `test_quickstart_needs_no_checkpoint_arguments_and_keeps_the_pin`
- [ ] `test_quickstart_requires_inputs_and_accepts_overrides`

### `test_recycle_depths.py` — PORT (6 tests)

- [ ] `test_full_trajectory_expands_whole_schedule`
- [ ] `test_partial_diffusion_takes_schedule_tail`
- [ ] `test_full_nsteps_defaults_to_nsteps`
- [ ] `test_schedule_length_must_match_full_nsteps`
- [ ] `test_no_schedule_uses_constant_depth_per_step_run`
- [ ] `test_schedule_requires_recycling_checkpoint`

### `test_skills.py` — PORT (4 tests)

- [ ] `test_skills_directory_is_non_empty`
- [ ] `test_skill_adapter_is_symlink_to_canonical_dir`
- [ ] `test_claude_md_includes_agents_md`
- [ ] `test_skill_is_discoverable_and_complete`

### `transforms/test_budget.py` — PORT (1 tests)

- [ ] `test_enforce_crop_budget`

### `transforms/test_conditions_and_modifiers.py` — PORT (20 tests)

- [ ] `test_condition_aliases`
- [ ] `test_default_annotation`
- [ ] `test_default_mask_no_annotation`
- [ ] `test_default_mask_with_annotation`
- [ ] `test_default_annotation`
- [ ] `test_default_mask_no_annotation`
- [ ] `test_default_mask_with_annotation`
- [ ] `test_default_annotations`
- [ ] `test_default_annotation`
- [ ] `test_default_mask_no_annotation`
- [ ] `test_default_mask_with_annotation`
- [ ] `test_rasa_calculation_and_binning`
- [ ] `test_non_ligand_neighbors_occlude_ligand_rasa`
- [ ] `test_rasa_no_negative_values`
- [ ] `test_ss_annotation_and_subset_validation`
- [ ] `test_gates_masks_and_clears_ground_truth`
- [ ] `test_clears_condition_when_no_partner_survives`
- [ ] `test_union_degrades_to_the_role_that_keeps_a_partner`
- [ ] `test_default_behavior`
- [ ] `test_group_assignment_by_residue_contiguity`

### `transforms/test_design_task_speed.py` — PORT (1 tests)

- [ ] `test_design_task_speed`

### `transforms/test_design_tasks.py` — PORT (12 tests)

- [ ] `test_unconditional_task_basic_properties`
- [ ] `test_all_simple_tasks_set_required_fields`
- [ ] `test_inverse_folding_conditions`
- [ ] `test_rna_inverse_folding_conditions`
- [ ] `test_structure_prediction_conditions`
- [ ] `test_dna_structure_context_conditions`
- [ ] `test_dna_sequence_context_conditions`
- [ ] `test_ppi_designs_protein_against_dna_target`
- [ ] `test_no_design_tasks_leave_atomized_things_without_sequence_conditioning`
- [ ] `test_unindexed_tokens_are_whole_and_atomized_atoms_stay_indexed`
- [ ] `test_coords_only_applied_to_sequence_conditioned_or_backbone_atoms`
- [ ] `test_design_task_modifiers`

### `transforms/test_double_bond_stereochemistry.py` — PORT (3 tests)

- [ ] `test_feature_represents_isomer_in_shared_frame`
- [ ] `test_ambiguous_geometry_is_not_encoded_or_scored_as_correct`
- [ ] `test_accuracy_and_overlapping_bond_frames`

### `transforms/test_metals_and_aromatic_dropout.py` — PORT (5 tests)

- [ ] `test_dropout_coordination_p_all_drops_every_coordination_bond`
- [ ] `test_dropout_coordination_p_all_zero_p_individual_zero_is_noop`
- [ ] `test_dropout_coordination_individual_drop_is_symmetric`
- [ ] `test_dropout_aromatic_p_one_zeros_all_aromatic_flags`
- [ ] `test_dropout_aromatic_p_zero_preserves_flags`

### `transforms/test_nucleic_hbonds.py` — PORT (1 tests)

- [ ] `test_base_pairs_from_real_dna_and_rna`

### `transforms/test_pi_ring_conditioning.py` — PORT (2 tests)

- [ ] `test_detected_rings_match_atomworks_ring_definitions`
- [ ] `test_subsampling_reveals_only_complete_rings`

### `transforms/test_protein_pocket_sampling.py` — PORT (1 tests)

- [ ] `test_protein_pocket_sampling`

### `transforms/test_secondary_structure_condition.py` — PORT (3 tests)

- [ ] `test_add_secondary_structure_condition_basic`
- [ ] `test_secondary_structure_requires_valid_dssp`
- [ ] `test_secondary_structure_island_sampling`

### `transforms/test_unindexing.py` — PORT (6 tests)

- [ ] `test_training_unindexing_preserves_internal_targets`
- [ ] `test_indexed_region_identical_across_modes`
- [ ] `test_appended_tokens_match_original_unindexed_atoms`
- [ ] `test_annotate_unindexed_insertion_reapplies_conditioning`
- [ ] `test_unindexing_preserves_source_condition_annotations`
- [ ] `test_annotate_unindexed_insertion_raises_on_conditioning_conflict`

### `transforms/test_unindexing_atom_matching.py` — PORT (1 tests)

- [ ] `test_atom_matching_with_name_priority_and_element_fallback`

### `utils/test_atom_permutation.py` — PORT (11 tests)

- [ ] `test_unknown_amino_acid_retains_observed_atom37_coordinates`
- [ ] `test_metrics_ignore_a_swap_of_indistinguishable_atoms`
- [ ] `test_a_real_difference_is_still_scored`
- [ ] `test_metrics_do_not_swap_across_a_subset_boundary`
- [ ] `test_permutation_on_a_conditioned_design`
- [ ] `test_a_batch_where_only_some_datasets_precomputed_permutations`
- [ ] `test_autoencoder_loss_ignores_a_relabelling_of_a_residue`
- [ ] `test_autoencoder_loss_ignores_a_relabelling_of_a_hard_ligand`
- [ ] `test_plinder_system_keeps_the_symmetry_it_was_chosen_for`
- [ ] `test_metrics_ignore_a_relabelling_of_a_whole_plinder_system`
- [ ] `test_batched_search_matches_a_per_group_loop`

### `utils/test_coordinate_transfer.py` — PORT (1 tests)

- [ ] `test_coordinate_transfer_preserves_annotations`

### `utils/test_structure_files.py` — PORT (4 tests)

- [ ] `test_structure_file_stem_preserves_dots`
- [ ] `test_structure_file_stem_rejects_unsupported_suffix`
- [ ] `test_unique_structure_files_rejects_sibling_encodings`
- [ ] `test_find_structure_file_missing_parent`

### `utils/test_subunit_permutation.py` — PORT (11 tests)

- [ ] `test_permutation_invariance`
- [ ] `test_tnfa_target_aligned_binder`
- [ ] `test_subunit_then_atom`
- [ ] `test_forbidden_swaps`
- [ ] `test_permutation_cap`
- [ ] `test_broadcast_per_model_annotations`
- [ ] `test_independent_residue_numbering`
- [ ] `test_independently_numbered_manganese_ions`
- [ ] `test_strict_correspondence`
- [ ] `test_small_and_empty_selections`
- [ ] `test_cropped_subunit_keeps_distinct_coverage`

### `utils/test_torch_utils.py` — PORT (4 tests)

- [ ] `test_minimum_loss_attempt_indices_uses_configured_losses`
- [ ] `test_minimum_loss_attempt_indices_rejects_unknown_loss`
- [ ] `test_map_to`
- [ ] `test_assert_no_nans`

### `utils/test_weight_loading.py` — PORT (6 tests)

- [ ] `test_custom_config`
- [ ] `test_pattern_match_policy`
- [ ] `test_basic_policies`
- [ ] `test_mixed_policies_and_fallbacks`
- [ ] `test_freeze_parameters_by_name_and_pattern`
- [ ] `test_load_weights_with_freezing`
