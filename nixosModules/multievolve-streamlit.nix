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
in
{
  options.services.multievolve-streamlit = {
    enable = lib.mkEnableOption "the MULTI-evolve Streamlit web application";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression "self.packages.${pkgs.stdenv.hostPlatform.system}.multievolve-streamlit";
      description = "Package that provides the multievolve-streamlit executable.";
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
      description = "Whether to create the configured service user and group.";
    };

    extraGroups = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [
        "video"
        "render"
      ];
      description = "Additional groups for the service user, for example GPU device access groups.";
    };

    workingDirectory = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/multievolve-streamlit";
      description = "Writable working directory for uploaded files, caches, and runtime outputs.";
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
        MULTIEVOLVE_ROOT = "/srv/multievolve";
      };
      description = "Additional environment variables for the service.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/multievolve.env";
      description = "Optional systemd EnvironmentFile for secrets or deployment-specific settings.";
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
    users.groups = lib.mkIf cfg.createUser {
      ${cfg.group} = { };
    };

    users.users = lib.mkIf cfg.createUser {
      ${cfg.user} = {
        isSystemUser = true;
        inherit (cfg) group extraGroups;
        home = toString cfg.workingDirectory;
        createHome = true;
      };
    };

    systemd.tmpfiles.rules = [
      "d ${toString cfg.workingDirectory} 0750 ${cfg.user} ${cfg.group} - -"
    ];

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    systemd.services.multievolve-streamlit = {
      description = "MULTI-evolve Streamlit web application";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        HOME = toString cfg.workingDirectory;
        MULTIEVOLVE_ROOT = toString cfg.workingDirectory;
        STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false";
      }
      // cfg.environment;

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = toString cfg.workingDirectory;
        ExecStart = lib.escapeShellArgs (
          [ "${cfg.package}/bin/multievolve-streamlit" ] ++ streamlitArgs ++ cfg.extraArgs
        );
        Restart = "on-failure";
        RestartSec = 5;
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ (toString cfg.workingDirectory) ];
      }
      // lib.optionalAttrs (cfg.environmentFile != null) { EnvironmentFile = cfg.environmentFile; };
    };
  };
}
