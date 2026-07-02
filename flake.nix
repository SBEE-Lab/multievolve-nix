{
  description = "MULTI-evolve — CUDA venv packaged with uv2nix-env";

  inputs = {
    # keep-sorted start
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    uv2nix-env.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix-env.inputs.treefmt-nix.follows = "treefmt-nix";
    uv2nix-env.url = "github:mulatta/uv2nix-env";
    # keep-sorted end
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      uv2nix-env,
    }:
    let
      inherit (nixpkgs) lib;

      # treefmt runs on every dev host; the venv is x86_64-linux only (the PyG
      # wheels are cp311/linux_x86_64).
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      eachSystem =
        f:
        lib.genAttrs systems (
          system:
          f {
            inherit system;
            pkgs = nixpkgs.legacyPackages.${system};
          }
        );

      treefmtEval = eachSystem (
        { pkgs, ... }:
        treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs = {
            deadnix.enable = true;
            keep-sorted.enable = true;
            nixfmt.enable = true;
            statix.enable = true;
          };
        }
      );

      linuxSystem = "x86_64-linux";
      pkgs = import nixpkgs {
        system = linuxSystem;
        config.allowUnfree = true;
      };

      # The repo root is the uv workspace: pyproject.toml + uv.lock + the
      # multievolve source (this fork's tree). torch/PyG/CUDA wheel fixups come
      # from uv2nix-env's universal wheel overlay; cusparselt is pinned to a good
      # wheel in pyproject. The only project-specific build override is fbpca's
      # missing build backend.
      inherit (uv2nix-env.lib) addBuildSystem;

      ws = uv2nix-env.lib.mkWorkspace {
        inherit pkgs;
        workspaceRoot = ./.;
        python = pkgs.python311;
        cuda = true;
        name = "multievolve";
        mainProgram = "multievolve";
        overrides = final: prev: {
          fbpca = addBuildSystem final { setuptools = [ ]; } prev.fbpca;
        };
      };
    in
    {
      # Pure, hash-locked venv with importable `multievolve` + its closure. Run the
      # pipeline scripts / Streamlit app as `${multievolve}/bin/python …`.
      packages.${linuxSystem} = rec {
        multievolve = ws.venv;
        multievolve-streamlit = ws.mkVenv {
          name = "multievolve-streamlit";
          mainProgram = "multievolve-streamlit";
        };
        default = multievolve;
      };

      devShells.${linuxSystem}.default = ws.mkDevShell {
        # git: complete-lock.py locates uv.lock via `git rev-parse --show-toplevel`.
        packages = [ pkgs.git ];
      };

      checks = eachSystem (
        { system, ... }:
        {
          formatting = treefmtEval.${system}.config.build.check self;
        }
        // lib.optionalAttrs (system == linuxSystem) { inherit (ws) venv; }
      );

      formatter = eachSystem ({ system, ... }: treefmtEval.${system}.config.build.wrapper);
    };
}
