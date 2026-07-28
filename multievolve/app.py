"""
Streamlit app for MULTI-evolve: A framework for engineering hyperactive multi-mutants.

This app provides an interactive web app to:
1. Train neural network models on protein mutation data
2. Propose optimized multi-mutant combinations
3. Generate MULTI-assembly mutagenic oligos for gene synthesis
4. Perform zero-shot predictions with protein language models
"""

import logging
import subprocess
import sys

import streamlit as st

from multievolve.app_io import (
    AppInputError,
    prepare_uploaded_inputs,
    validate_identifier,
)

logger = logging.getLogger(__name__)

_IDENTIFIER_HELP = (
    "Start with a letter or number and use only letters, numbers, '.', '_', or '-'."
)

def setup_page():
    """Configure basic Streamlit page settings"""
    st.set_page_config(
        page_title="MULTI-evolve",
        page_icon="🧬",
        layout="wide"
    )

    # Source-controlled static markup only; runtime output is rendered with st.code.
    st.markdown("""
        <h1 style="margin-bottom: 0; padding-bottom: 0;">MULTI-evolve</h1>
        <p style="color: #666; margin-top: 0.2rem; font-size: 1.05rem;">
            A framework for engineering hyperactive multi-mutants
        </p>
        <hr style="margin-top: 0.5rem; margin-bottom: 0.5rem; border: none; border-top: 1px solid #e0e0e0;">
    """, unsafe_allow_html=True)

    # Source-controlled global styles — injected once, available to all tabs
    st.markdown("""
        <style>
            /* Reduce default top padding */
            .block-container {
                padding-top: 2rem;
            }

            /* Compact file uploader drop zones with breathing room */
            [data-testid="stFileUploader"] section {
                padding: 0.4rem 0.75rem;
            }

            /* Add inner padding to form containers */
            [data-testid="stForm"] {
                padding: 0.75rem 1rem;
            }

            /* Tab styling — bordered pill labels */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 0;
            }
            .stTabs [data-baseweb="tab"] {
                border: 1px solid #e0e0e0;
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                padding: 0.5rem 1rem;
                background-color: #f9f9f9;
            }
            .stTabs [aria-selected="true"] {
                background-color: #ffffff;
                border-bottom: 2px solid #ffffff;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

def validate_files(
    protein_name,
    *,
    wt_files_aa=None,
    wt_file_aa=None,
    wt_file_dna=None,
    dataset_file=None,
    mutations_file=None,
    pdb_files=None,
):
    """Validate and persist browser uploads, returning canonical trusted paths."""
    try:
        return prepare_uploaded_inputs(
            protein_name,
            wt_files_aa=wt_files_aa,
            wt_file_aa=wt_file_aa,
            wt_file_dna=wt_file_dna,
            dataset_file=dataset_file,
            mutations_file=mutations_file,
            pdb_files=pdb_files,
        )
    except AppInputError as exc:
        st.error(str(exc))
        return None
    except Exception:
        logger.exception("Unexpected failure while preparing uploaded files")
        st.error("Unexpected upload failure. Check the service logs for details.")
        return None


def validate_name(value, label):
    """Display an actionable error for an invalid artifact identifier."""
    try:
        return validate_identifier(value, label)
    except AppInputError as exc:
        st.error(str(exc))
        return None


def render_terminal_output(placeholder, output):
    """Render child-process output as text rather than browser-interpreted HTML."""
    placeholder.code(
        output,
        language=None,
        wrap_lines=True,
        height=400,
    )


def stream_command_output(command, placeholder):
    """Run one argv-based child process while streaming its combined output."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Child process did not provide an output stream")

    output_lines = []
    for line in iter(process.stdout.readline, ""):
        output_lines.append(line.rstrip())
        render_terminal_output(placeholder, "\n".join(output_lines))

    return_code = process.wait()
    render_terminal_output(placeholder, "\n".join(output_lines))
    return return_code

def train_models():
    """Train neural network models section"""

    with st.form("train_models_form"):
        col1, col2 = st.columns([2,3])

        with col1:
            protein_name = st.text_input("Protein Name", help=_IDENTIFIER_HELP)
            wt_files_aa = st.file_uploader("Upload Wildtype Amino Acid Sequence FASTA", accept_multiple_files=True, type=['fasta', 'fa'])
            dataset_file = st.file_uploader("Upload Training Dataset (CSV)", type=['csv'], accept_multiple_files=False)
            st.divider()
            experiment_name = st.text_input(
                "Experiment Name", value="test", help=_IDENTIFIER_HELP
            )
            mode = st.selectbox("Training Mode", ["test", "standard"])
            seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=42)
            split_seed = st.number_input("Split Seed", min_value=0, max_value=2**32 - 1, value=42)
            cv_folds = st.number_input("CV Folds", min_value=2, value=5)
            device = st.selectbox("Training Device", ["auto", "cpu", "cuda"])
            deterministic = st.checkbox("Deterministic Mode", value=False)

        with col2:
            st.markdown("""
            ### Step 1: Train Neural Network Models

            This tool performs a grid search over many neural network architectures to find the best performing model for a given protein and dataset.
            """)
            with st.expander("Input Files and Parameters", expanded=False):
                st.markdown("""
                - **Training Dataset (CSV)**: CSV file with columns 'mutation' and 'property_value'. Variants should be formatted as ```A40P/E61Y```, or for protein complexes as ```A40P/E61Y:WT```, where ```:``` separates the individual chains (e.g. ```chain 1 mutations:chain 2 mutations```), ```/``` separates the individual mutations, and ```WT``` indicates the wildtype sequence. A sample dataset for APEX peroxidase can be found in ```data/example_protein/example_dataset.csv```. For a protein complex example, use ```data/example_multichain_protein/example_dataset.csv```.
                - **Wildtype Amino Acid Sequence FASTA**: Protein sequence in FASTA format. Upload multiple sequence files if working with a protein complex in the same order as you formatted the variants in the training dataset. A sample sequence of APEX peroxidase can be found in ```data/example_protein/apex.fasta```. For a protein complex example, upload in the following order: ```data/example_multichain_protein/vh_chain1.fasta```, ```data/example_multichain_protein/vl_chain2.fasta```.
                - **Experiment Name**: Name of the model training experiment (e.g. APEX_gridsearch). This should be used for the subsequent step 2 for proposing mutations.
                - **Training Mode**:
                    - `test`: Test the training process for a single architecture.
                    - `standard`: Performs a grid search over many architectures. Will take a longer time to run.
                - **Seed/Split Seed**: Base model seed and fold-assignment seed.
                - **CV Folds**: Number of architecture-selection cross-validation folds (default: 5).
                - **Training Device**: `auto`, `cpu`, or required `cuda`.
                - **Deterministic Mode**: Requests deterministic PyTorch algorithms; unsupported operations fail explicitly.
                - Completed fold/config jobs are checksummed and reused when the same experiment is resumed. Missing or modified aggregate results are rebuilt from valid jobs. Signed property values are supported.
                """)

        submitted = st.form_submit_button("Train Models", type="primary")

    if submitted:
        if not all([experiment_name, protein_name, wt_files_aa, dataset_file]):
            st.error("Please fill in all required fields")
            return

        experiment_name = validate_name(experiment_name, "Experiment name")
        if experiment_name is None:
            return
        prepared = validate_files(
            protein_name,
            wt_files_aa=wt_files_aa,
            dataset_file=dataset_file,
        )
        if prepared is None:
            return

        try:
            wt_paths = prepared.wt_files_aa
            dataset_path = prepared.dataset_file

            # Show the command that will be executed
            command = [
                sys.executable, "-m", "multievolve.cli.train",
                "--experiment-name", experiment_name,
                "--protein-name", protein_name,
                "--wt-files", ",".join(str(wt_path) for wt_path in wt_paths),
                "--training-dataset-fname", str(dataset_path),
                "--mode", mode,
                "--seed", str(seed),
                "--split-seed", str(split_seed),
                "--cv-folds", str(cv_folds),
                "--device", device,
            ]
            if deterministic:
                command.append("--deterministic")

            st.subheader("Terminal Output:")
            st.code(f"$ {' '.join(command)}", language="bash")

            with st.container(border=True):
                terminal_output = st.empty()

            with st.spinner("Training models..."):
                return_code = stream_command_output(command, terminal_output)

                if return_code == 0:
                    st.success("✅ Model training completed successfully!")
                else:
                    st.error(f"❌ Training failed with exit code: {return_code}")

        except Exception:
            logger.exception("Unexpected failure while training models")
            st.error("Unexpected training failure. Check the service logs for details.")

def propose_mutations():
    """Propose mutations section"""

    with st.form("propose_mutations_form"):
        col1, col2 = st.columns([2,3])

        with col1:
            protein_name = st.text_input(
                "Protein Name", key="propose_protein", help=_IDENTIFIER_HELP
            )
            wt_files_aa = st.file_uploader("Upload Wildtype Amino Acid Sequence FASTA", accept_multiple_files=True, type=['fasta', 'fa'], key="propose_wt")
            dataset_file = st.file_uploader("Upload Training Dataset (CSV)", type=['csv'], key="propose_dataset")
            mutation_pool = st.file_uploader("Upload Mutation Pool (CSV)", type=['csv'])
            st.divider()
            experiment_name = st.text_input(
                "Experiment Name", key="propose_exp", help=_IDENTIFIER_HELP
            )
            min_mutations = st.number_input("Minimum Mutations", min_value=2, value=3)
            max_mutations = st.number_input("Maximum Mutations", min_value=2, value=10)
            top_muts = st.number_input("Top Mutations per Load", min_value=1, value=3)
            max_candidates = st.number_input("Maximum Candidates", min_value=1, value=100000)
            seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=42, key="propose_seed")
            split_seed = st.number_input("Split Seed", min_value=0, max_value=2**32 - 1, value=42, key="propose_split_seed")
            ensemble_folds = st.number_input("Ensemble Folds", min_value=2, value=10)
            device = st.selectbox("Training Device", ["auto", "cpu", "cuda"], key="propose_device")
            deterministic = st.checkbox("Deterministic Mode", value=False, key="propose_deterministic")
            export_name = st.text_input(
                "Export Name",
                value="multievolve_proposals",
                help=_IDENTIFIER_HELP,
            )

        with col2:
            st.markdown("""
            ### Step 2: Propose MULTI-evolve Variants

            This tool proposes MULTI-evolve variants using a trained neural network model, whose ideal architecture is selected from a grid search in Step 1.
            """)
            with st.expander("Input Files and Parameters", expanded=False):
                st.markdown("""
                - **Wildtype Amino Acid Sequence FASTA**: Protein sequence in FASTA format. Upload multiple sequence files if working with a protein complex. Same file(s) as Step 1. A sample sequence of APEX peroxidase can be found in ```data/example_protein/apex.fasta```. For a protein complex example, upload in the following order: ```data/example_multichain_protein/vh_chain1.fasta```, ```data/example_multichain_protein/vl_chain2.fasta```.
                - **Training Dataset (CSV)**: CSV file with columns 'mutation' and 'property_value'. Same file as Step 1. A sample dataset for APEX peroxidase can be found in ```data/example_protein/example_dataset.csv```. For a protein complex example, use ```data/example_multichain_protein/example_dataset.csv```.
                - **Mutation Pool (CSV)**: Path to the mutation pool CSV file, which is a list of mutations to be used to generate the proposed combinatorial variants. It is a one column no header CSV file. Example is provided in ```data/example_protein/combo_muts.csv```. For a protein complex example, use ```data/example_multichain_protein/combo_muts.csv```.
                - **Experiment Name**: Name of the model training experiment (e.g. APEX_gridsearch). Same experiment name as Step 1.
                - **Minimum/Maximum Mutations**: Inclusive mutational-load range to generate. The CLI retains the historical 3–10 defaults; the paper's experiments support prioritizing variants with at most 7 substitutions.
                - **Top Mutations per Load**: Number of top mutations to propose per mutational load.
                - **Maximum Candidates**: Safety limit checked before final model training and combinatorial candidate generation.
                - **Seed/Split Seed**: Base ensemble-model seed and fold-assignment seed.
                - **Ensemble Folds**: Number of final models retrained with the selected architecture (default: 10).
                - **Training Device**: `auto`, `cpu`, or required `cuda`.
                - **Deterministic Mode**: Requests deterministic PyTorch algorithms; unsupported operations fail explicitly.
                - Step 2 requires the same canonical dataset, WT FASTA, feature, split seed, compatible software, and artifact schema as Step 1. Completed ensemble folds are hash-validated and reused. Predictions remain in original property units, including for signed assays.
                - **Export Name**: Name of the exported csv file containing the list of the proposed variants. This csv file can be used to generate MULTI-assembly mutagenic oligos for cloning the proposed variants in the ```Design MULTI-assembly Oligos``` tab.
                """)
            with st.expander("Outputs", expanded=False):
                st.markdown("""
                A CSV file will be generated:
                - `<Export Name>.csv`: List of proposed variants. If it is a protein complex, it will export files for each chain (e.g. ```<Export Name>_chain_1_mutants.csv```)
                """)

        submitted = st.form_submit_button("Propose Mutations", type="primary")

    if submitted:
        if not all([experiment_name, protein_name, wt_files_aa, dataset_file, mutation_pool, export_name]):
            st.error("Please fill in all required fields")
            return
        if min_mutations > max_mutations:
            st.error("Minimum Mutations must be less than or equal to Maximum Mutations")
            return

        experiment_name = validate_name(experiment_name, "Experiment name")
        export_name = validate_name(export_name, "Export name")
        if experiment_name is None or export_name is None:
            return
        prepared = validate_files(
            protein_name,
            wt_files_aa=wt_files_aa,
            dataset_file=dataset_file,
            mutations_file=mutation_pool,
        )
        if prepared is None:
            return

        try:
            wt_paths = prepared.wt_files_aa
            dataset_path = prepared.dataset_file
            mutation_pool_path = prepared.mutations_file

            # Show the command that will be executed
            command = [
                sys.executable, "-m", "multievolve.cli.propose",
                "--experiment-name", experiment_name,
                "--protein-name", protein_name,
                "--wt-files", ",".join(str(wt_path) for wt_path in wt_paths),
                "--training-dataset", str(dataset_path),
                "--mutation-pool", str(mutation_pool_path),
                "--min-mutations", str(min_mutations),
                "--max-mutations", str(max_mutations),
                "--top-muts-per-load", str(top_muts),
                "--max-candidates", str(max_candidates),
                "--seed", str(seed),
                "--split-seed", str(split_seed),
                "--ensemble-folds", str(ensemble_folds),
                "--device", device,
                "--export-name", export_name
            ]
            if deterministic:
                command.append("--deterministic")

            st.subheader("Terminal Output:")
            st.code(f"$ {' '.join(command)}", language="bash")

            with st.container(border=True):
                terminal_output = st.empty()

            with st.spinner("Proposing mutations..."):
                return_code = stream_command_output(command, terminal_output)

                if return_code == 0:
                    st.success("✅ MULTI-evolve variants proposed successfully!")
                else:
                    st.error(
                        f"❌ MULTI-evolve variants proposal failed with exit code: {return_code}"
                    )

        except Exception:
            logger.exception("Unexpected failure while proposing mutations")
            st.error("Unexpected proposal failure. Check the service logs for details.")

def design_oligos():
    """Design MULTI-assembly oligos section"""

    with st.form("design_oligos_form"):
        col1, col2 = st.columns([2,3])

        with col1:
            protein_name = st.text_input(
                "Protein Name",
                key="MULTI-assembly_protein",
                help=_IDENTIFIER_HELP,
            )
            wt_file_dna = st.file_uploader("Upload Wildtype DNA Sequence FASTA", type=['fasta', 'fa'], key="oligo_wt")
            mutations_file = st.file_uploader("Upload Mutations File (CSV)", type=['csv'])
            st.divider()
            species = st.selectbox("Species", ["human", "ecoli", "yeast"])
            tm = st.number_input("Melting Temperature (°C)", value=80.0)
            overhang = st.number_input("Overhang Length", value=33)
            oligo_direction = st.selectbox("Oligo Direction", ["top", "bottom"])
            output_type = st.selectbox("Output Type", ["design", "update"])

        with col2:
            st.markdown("""
            ### Step 3: Generate MULTI-assembly Mutagenic Oligos

            This tool generates mutagenic oligos for MULTI-assembly cloning of multi-mutant variants.
            """)
            with st.expander("Input Files and Parameters", expanded=False):
                st.markdown("""
                - **Protein Name**: Name of the protein to generate oligos for.
                - **Wildtype DNA Sequence FASTA**: DNA sequence of the wildtype protein with overhangs from the protein's MULTI-assembly vector. The sequence should include overhangs for the MULTI-assembly oligos, wherein the overhangs are the same length on both ends of the DNA sequence. Recommended overhang length is 33 bp or longer. An example is found in ```data/example_protein/APEX_33overhang.fasta```.
                - **Mutations File (CSV)**: List of proposed variants to generate oligos for. It is a one column no-header csv file with the variants. See ```data/example_protein/MULTI-assembly_input.csv``` for an example of the csv format.
                - **Species**: Codon usage table selection (human/ecoli/yeast).
                - **Melting Temperature**: Target Tm for oligos (recommended: 80°C).
                - **Overhang Length**: Length of overhangs on both ends of sequence.
                - **Oligo Direction**:
                    - `top`: Oligos bind 5' to 3' in top strand orientation.
                    - `bottom`: Oligos bind 3' to 5' in bottom strand orientation.
                - **Output Type**:
                    - `design`: Generate new oligo designs.
                    - `update`: Update existing oligo IDs.
                """)
            with st.expander("Outputs", expanded=False):
                st.markdown("""
                Two CSV files will be generated:
                1. `cloning_sheet.csv`: Assembly instructions describing which oligos to pool for each variant.
                2. `oligos.csv`: Oligo sequences and IDs.

                The oligo IDs in `oligos.csv` can be customized with user-defined IDs. After editing the `oligos.csv` file, re-running with `update` will sync IDs between files.
                """)

        submitted = st.form_submit_button("Design Oligos", type="primary")

    if submitted:
        if not all([protein_name, mutations_file, wt_file_dna]):
            st.error("Please fill in all required fields")
            return

        prepared = validate_files(
            protein_name,
            wt_file_dna=wt_file_dna,
            mutations_file=mutations_file,
        )
        if prepared is None:
            return

        try:
            mutations_path = prepared.mutations_file
            wt_path = prepared.wt_file_dna

            # Show the command that will be executed
            command = [
                sys.executable, "-m", "multievolve.cli.assembly_design",
                "--mutations-file", str(mutations_path),
                "--wt-fasta", str(wt_path),
                "--overhang", str(overhang),
                "--species", species,
                "--oligo-direction", oligo_direction,
                "--tm", str(tm),
                "--output", output_type
            ]

            st.subheader("Terminal Output:")
            st.code(f"$ {' '.join(command)}", language="bash")

            with st.container(border=True):
                terminal_output = st.empty()

            with st.spinner("Designing oligos..."):
                return_code = stream_command_output(command, terminal_output)

                if return_code == 0:
                    st.success("✅ Oligo design completed successfully!")
                else:
                    st.error(f"❌ Oligo design failed with exit code: {return_code}")

        except Exception:
            logger.exception("Unexpected failure while designing oligos")
            st.error("Unexpected oligo-design failure. Check the service logs for details.")

def zeroshot_predictions():
    """Perform zero-shot predictions section"""

    with st.form("zeroshot_predictions_form"):
        col1, col2 = st.columns([2,3])

        with col1:
            protein_name = st.text_input(
                "Protein Name", key="zeroshot_protein", help=_IDENTIFIER_HELP
            )
            wt_file_aa = st.file_uploader("Upload Wildtype Amino Acid Sequence FASTA", type=['fasta', 'fa'], key="zeroshot_wt")
            pdb_files = st.file_uploader("Upload PDB/CIF Files", type=['pdb', 'cif'], accept_multiple_files=True, key="zeroshot_pdb")
            st.divider()
            chain_id = st.text_input("Chain ID", value="A", key="zeroshot_chain")
            variants = st.number_input("Number of Variants", min_value=1, value=24)
            excluded_pos = st.text_input("Excluded Positions (comma-separated, optional)", value="1,10,30", key="zeroshot_excluded")
            norm_method = st.selectbox("Normalizing Method", ["aa_substitution_type", "aa_mutation"], key="zeroshot_norm")

        with col2:
            st.markdown("""
            ### Protein Language Model Zero-shot Ensemble

            This tool performs zero-shot predictions with a protein language model ensemble to nominate mutations.
            """)
            with st.expander("Input Files and Parameters", expanded=False):
                st.markdown("""
                - **Wildtype Amino Acid Sequence FASTA**: Protein sequence in FASTA format.
                - **PDB/CIF Files**: One or more structure files in PDB or CIF format. Provide multiple structure files if there are different models (e.g. top 5 predicted structures from AlphaFold).
                - **Chain ID**: Chain ID of the targeted protein in the structure files.
                - **Number of Variants**: Number of variants to nominate per method (default: 24)
                - **Excluded Positions**: Comma-separated list of positions to exclude from mutation (e.g. 1,5,20). Leave empty if no positions should be excluded.
                - **Normalizing Method**: Method for normalizing fold-change scores:
                    - `aa_substitution_type`: Group by specific amino acid substitution type (e.g. all alanine to proline mutations, A→P mutations).
                    - `aa_mutation`: Group by amino acid mutation (e.g. all mutations to proline, →P).
                """)
            with st.expander("Outputs", expanded=False):
                st.markdown("""
                A CSV file will be generated:
                - `plm_zeroshot_ensemble_nominated_mutations.csv`: List of proposed variants and nominating methods.
                """)

        submitted = st.form_submit_button("Run Zero-shot Predictions", type="primary")

    if submitted:
        if not all([protein_name, wt_file_aa, pdb_files, chain_id]):
            st.error("Please fill in all required fields")
            return

        prepared = validate_files(
            protein_name,
            wt_file_aa=wt_file_aa,
            pdb_files=pdb_files,
        )
        if prepared is None:
            return

        try:
            wt_path = prepared.wt_file_aa
            pdb_paths = prepared.pdb_files

            # Show the command that will be executed
            command = [
                sys.executable, "-m", "multievolve.cli.plm_zeroshot_ensemble",
                "--wt-file", str(wt_path),
                "--pdb-files", ",".join(str(path) for path in pdb_paths),
                "--chain-id", chain_id,
                "--variants", str(variants),
                "--normalizing-method", norm_method
            ]

            # Only add excluded-positions flag if it's provided and not empty
            if excluded_pos.strip():
                command.extend(["--excluded-positions", excluded_pos])

            st.subheader("Terminal Output:")
            st.code(f"$ {' '.join(command)}", language="bash")

            with st.container(border=True):
                terminal_output = st.empty()

            with st.spinner("Running Zero-shot Predictions..."):
                return_code = stream_command_output(command, terminal_output)

                if return_code == 0:
                    st.success("✅ Zero-shot predictions completed successfully!")
                else:
                    st.error(
                        f"❌ Zero-shot predictions failed with exit code: {return_code}"
                    )

        except Exception:
            logger.exception("Unexpected failure while running zero-shot predictions")
            st.error("Unexpected zero-shot failure. Check the service logs for details.")

def about():
    st.markdown("""

    This tool provides an interactive web app to perform the computational steps of MULTI-evolve (model-guided, universal, targeted installation of multi-mutants), an end-to-end framework for efficiently engineering hyperactive multi-mutants.

    The interactive web app has the following uses:
    1. Implement the MULTI-evolve framework to propose multi-mutants and generate the associated MULTI-assembly mutagenic oligos for gene synthesis:

        (a) Train fully connected neural networks to predict the fitness of a given sequence.

        (b) Choose the best performing neural network and use it to predict combinatorial variants.

        (c) For the chosen multi-mutants, generate the MULTI-assembly mutagenic oligos for gene synthesis.

    2. Perform the Protein Language Model Zero-shot Ensemble Approach used in the MULTI-evolve framework.
    """)

def file_locations():
    st.markdown("### Where are my files saved?")
    st.markdown("""
    When you run any of the tools above, MULTI-evolve automatically creates a folder called **`proteins`**
    (inside the main MULTI-evolve directory) to keep all of your work organized. Inside that folder,
    each protein you work with gets its own sub-folder named after the **Protein Name** you entered.

    Here is what that folder looks like and what each part contains:
    """)

    st.code("""
proteins/
└── your_protein_name/
    │
    ├── your_uploaded_files          ← Your uploaded FASTA, CSV, and structure files
    │
    ├── feature_cache/               ← Saved sequence features (speeds up re-runs)
    │
    ├── model_cache/                 ← Trained models and comparison results
    │   └── your_dataset_name/
    │       ├── objects/             ← The saved trained models
    │       └── results/             ← Performance metrics from model comparisons
    │
    ├── proposers/                   ← Proposed multi-mutant variants
    │   └── results/                 ← CSV files with proposed variants and scores
    │
    ├── split_cache/                 ← Saved data splits (train/test sets)
    │
    ├── cloning_sheet.csv            ← Which oligos to pool for each variant
    ├── oligos.csv                   ← Oligo sequences for ordering
    └── plm_zeroshot_ensemble_nominated_mutations.csv ← Nominated mutations from PLM ensemble
    """, language=None)

    st.markdown("#### Quick guide to finding your results")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**After Step 1 (Train Models)**\n\n"
            "Your trained models are saved in:\n\n"
            "`proteins/your_protein_name/model_cache/`"
        )
        st.info(
            "**After Step 3 (Generate Oligos)**\n\n"
            "Your oligo designs are saved in:\n\n"
            "`proteins/your_protein_name/cloning_sheet.csv`\n\n"
            "`proteins/your_protein_name/oligos.csv`"
        )

    with col2:
        st.info(
            "**After Step 2 (Propose Multi-mutants)**\n\n"
            "`proteins/your_protein_name/proposers/results/` — Predictions for all multi-mutant variants\n\n"
            "`proteins/your_protein_name/multievolve_proposals.csv` — The proposed variants to test\n\n"
        )
        st.info(
            "**After PLM Zero-shot Ensemble**\n\n"
            "Your nominated mutations are saved in:\n\n"
            "`proteins/your_protein_name/plm_zeroshot_ensemble_nominated_mutations.csv`"
        )

def main():
    """Main function to run the Streamlit app"""
    setup_page()

    # Create tabs for different functionalities
    tab5, tab1, tab2, tab3, tab4, tab6 = st.tabs([
        "About",
        "Train Models",
        "Propose Multi-mutants",
        "Generate MULTI-assembly Oligos",
        "Perform PLM Zero-shot Ensemble",
        "Output Files",
    ])

    with tab5:
        about()

    with tab1:
        train_models()

    with tab2:
        propose_mutations()

    with tab3:
        design_oligos()

    with tab4:
        zeroshot_predictions()

    with tab6:
        file_locations()



if __name__ == "__main__":
    main()
