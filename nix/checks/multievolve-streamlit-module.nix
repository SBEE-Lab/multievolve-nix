{
  pkgs,
  streamlitModule,
  ...
}:

import ../nixosTests/multievolve-streamlit.nix {
  inherit pkgs;
  module = streamlitModule;
}
