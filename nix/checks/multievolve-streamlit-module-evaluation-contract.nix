{
  nixpkgs,
  pkgs,
  streamlitModule,
  streamlitPackage,
  system,
  ...
}:

let
  inherit (nixpkgs) lib;

  invalidModuleConfiguration = lib.nixosSystem {
    inherit system;
    modules = [
      streamlitModule
      {
        system.stateVersion = "26.11";
        services.multievolve-streamlit = {
          enable = true;
          package = streamlitPackage;
          environment.HOME = "/wrong/home";
          workingDirectory = "/tmp/multievolve";
        };
      }
    ];
  };
  failedModuleAssertions = builtins.filter (
    assertion: !assertion.assertion
  ) invalidModuleConfiguration.config.assertions;
in
assert lib.any (
  assertion: lib.hasInfix "environment may not override HOME" assertion.message
) failedModuleAssertions;
assert lib.any (
  assertion: lib.hasInfix "workingDirectory must be a dedicated writable path" assertion.message
) failedModuleAssertions;
pkgs.runCommand "multievolve-streamlit-module-evaluation-contract" { } ''
  touch "$out"
''
