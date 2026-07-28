{
  pkgs,
  streamlitPackage,
  ...
}:

pkgs.runCommand "multievolve-streamlit-package-contract" { } ''
  export HOME="$TMPDIR"
  timeout=${pkgs.coreutils}/bin/timeout
  $timeout 120 ${streamlitPackage}/bin/multievolve-streamlit --help >/dev/null
  $timeout 120 ${streamlitPackage}/bin/python -m multievolve.cli.train --help >/dev/null
  $timeout 120 ${streamlitPackage}/bin/python -m multievolve.cli.propose --help >/dev/null
  $timeout 120 ${streamlitPackage}/bin/python -m multievolve.cli.assembly_design --help >/dev/null
  $timeout 120 ${streamlitPackage}/bin/python -m multievolve.cli.plm_zeroshot_ensemble --help >/dev/null
  touch "$out"
''
