{ self }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.multievolve-streamlit;
  system = pkgs.stdenv.hostPlatform.system;
  defaultPackage =
    self.packages.${system}.multievolve-streamlit
      or (throw "MULTI-evolve is currently packaged only for x86_64-linux; no multievolve-streamlit package for ${system}");

  reservedEnvironmentVariables = [
    "HOME"
    "MULTIEVOLVE_ROOT"
  ];
  overriddenReservedVariables = builtins.filter (
    name: builtins.hasAttr name cfg.environment
  ) reservedEnvironmentVariables;

  stateRoot = toString cfg.workingDirectory;
  inaccessibleStateRootPrefixes = [
    builtins.storeDir
    "/home"
    "/root"
    "/run/user"
    "/tmp"
    "/var/tmp"
  ];
  inaccessibleStateRoot = lib.findFirst (
    prefix: stateRoot == prefix || lib.hasPrefix "${prefix}/" stateRoot
  ) null inaccessibleStateRootPrefixes;

  streamlitArgs = [
    "--server.headless"
    "true"
    "--server.address"
    cfg.host
    "--server.port"
    (toString cfg.port)
    "--server.fileWatcherType"
    "none"
  ];

  command = [
    "${pkgs.coreutils}/bin/env"
    "HOME=${stateRoot}"
    "MULTIEVOLVE_ROOT=${stateRoot}"
    "${cfg.package}/bin/multievolve-streamlit"
  ]
  ++ streamlitArgs
  ++ cfg.extraArgs;
in
{
  options.services.multievolve-streamlit = {
    enable = lib.mkEnableOption "the MULTI-evolve Streamlit web application";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression "self.packages.${pkgs.stdenv.hostPlatform.system}.multievolve-streamlit";
      description = ''
        Package that provides the multievolve-streamlit executable and a Python
        environment containing the multievolve.cli modules launched by the app.
      '';
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "multievolve";
      description = "User account under which the service runs.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "multievolve";
      description = "Group under which the service runs.";
    };

    createUser = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to create the configured service user and group. When disabled,
        both accounts must exist before tmpfiles and the service are started.
      '';
    };

    extraGroups = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [
        "video"
        "render"
      ];
      description = ''
        Supplementary groups for the service process, for example GPU device
        access groups. They must already exist when createUser is disabled.
      '';
    };

    workingDirectory = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/multievolve-streamlit";
      description = ''
        Writable state root for uploaded files, caches, and runtime outputs.
        The service working directory, HOME, and MULTIEVOLVE_ROOT are all bound
        to this dedicated path. It must remain visible with ProtectHome and
        PrivateTmp enabled and cannot be located in the Nix store.
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address on which Streamlit listens.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8501;
      description = "Port on which Streamlit listens.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Whether to open the configured TCP port in the NixOS firewall.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = {
        OMP_NUM_THREADS = "1";
      };
      description = ''
        Additional environment variables for the service. HOME and
        MULTIEVOLVE_ROOT are reserved; set workingDirectory to change the
        service state root.
      '';
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/etc/multievolve.env";
      description = ''
        Optional systemd EnvironmentFile for deployment-specific settings.
        Environment variables are not a secure secret transport. HOME and
        MULTIEVOLVE_ROOT are always reset to workingDirectory when the service
        process starts.
      '';
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [
        "--server.baseUrlPath"
        "/multievolve"
      ];
      description = "Additional arguments passed to streamlit run.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = overriddenReservedVariables == [ ];
        message = ''
          services.multievolve-streamlit.environment may not override ${lib.concatStringsSep ", " overriddenReservedVariables}; set services.multievolve-streamlit.workingDirectory instead
        '';
      }
      {
        assertion = stateRoot != "/" && inaccessibleStateRoot == null;
        message = ''
          services.multievolve-streamlit.workingDirectory must be a dedicated writable path outside /home, /root, /run/user, /tmp, /var/tmp, and the Nix store
        '';
      }
    ];

    users.groups = lib.mkIf cfg.createUser {
      ${cfg.group} = { };
    };

    users.users = lib.mkIf cfg.createUser {
      ${cfg.user} = {
        isSystemUser = true;
        inherit (cfg) group extraGroups;
        home = stateRoot;
        createHome = true;
      };
    };

    systemd.tmpfiles.settings."10-multievolve-streamlit".${stateRoot}.d = {
      mode = "0750";
      inherit (cfg) user;
      inherit (cfg) group;
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    systemd.services.multievolve-streamlit = {
      description = "MULTI-evolve Streamlit web application";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = cfg.environment // {
        HOME = stateRoot;
        MULTIEVOLVE_ROOT = stateRoot;
        STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false";
      };

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = stateRoot;
        ExecStart = lib.escapeShellArgs command;
        Restart = "on-failure";
        RestartSec = 5;
        CapabilityBoundingSet = "";
        LockPersonality = true;
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectControlGroups = true;
        ProtectHome = true;
        ProtectKernelLogs = true;
        ProtectKernelModules = true;
        ProtectKernelTunables = true;
        ProtectSystem = "strict";
        ReadWritePaths = [ stateRoot ];
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        SupplementaryGroups = cfg.extraGroups;
        # Keep artifacts readable by the explicitly configured service group.
        UMask = "0027";
      }
      // lib.optionalAttrs (cfg.environmentFile != null) { EnvironmentFile = cfg.environmentFile; };
    };
  };
}
