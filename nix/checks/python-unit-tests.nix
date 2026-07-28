{
  pkgs,
  self,
  streamlitPackage,
  ...
}:

pkgs.runCommand "multievolve-python-unit-tests" { } ''
  export HOME="$TMPDIR"
  export MULTIEVOLVE_ROOT="$TMPDIR"
  export MPLCONFIGDIR="$TMPDIR/matplotlib"
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  cd ${self.outPath}
  ${streamlitPackage}/bin/python -m unittest discover -s tests -v
  touch "$out"
''
