{
  module,
  pkgs,
}:
let
  testPackage = pkgs.writeShellApplication {
    name = "multievolve-streamlit";
    runtimeInputs = [ pkgs.python3 ];
    text = ''
      address=127.0.0.1
      port=8501
      original_args=("$@")
      while (( $# > 0 )); do
        case "$1" in
          --server.address)
            address=$2
            shift 2
            ;;
          --server.port)
            port=$2
            shift 2
            ;;
          *)
            shift
            ;;
        esac
      done

      exec python3 - "$address" "$port" "''${original_args[@]}" <<'PY'
      import errno
      import grp
      import json
      import os
      import pathlib
      import sys
      from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

      address, port, *arguments = sys.argv[1:]
      root = pathlib.Path(os.environ["MULTIEVOLVE_ROOT"])
      assert pathlib.Path(os.environ["HOME"]) == root
      assert root.is_dir()
      (root / "service-started").write_text("ok\n")
      (root / "arguments.json").write_text(json.dumps(arguments))

      sandbox_control = pathlib.Path("/srv/sandbox-control")
      assert sandbox_control.is_dir()
      try:
          (sandbox_control / "probe").write_text("unexpected\n")
      except OSError as error:
          assert error.errno == errno.EROFS
          (root / "sandbox-protected").write_text("ok\n")
      else:
          raise RuntimeError("ProtectSystem did not make /srv read-only")

      class Handler(BaseHTTPRequestHandler):
          def do_GET(self):
              body = json.dumps({
                  "home": os.environ["HOME"],
                  "root": os.environ["MULTIEVOLVE_ROOT"],
                  "file_var": os.environ.get("FILE_VAR"),
                  "supplementary_groups": sorted(
                      grp.getgrgid(group_id).gr_name for group_id in os.getgroups()
                  ),
              }).encode()
              self.send_response(200)
              self.send_header("Content-Type", "application/json")
              self.send_header("Content-Length", str(len(body)))
              self.end_headers()
              self.wfile.write(body)

          def log_message(self, format, *args):
              pass

      ThreadingHTTPServer((address, int(port)), Handler).serve_forever()
      PY
    '';
  };
in
pkgs.testers.runNixOSTest {
  name = "multievolve-streamlit";

  nodes = {
    machine = {
      imports = [ module ];

      environment.etc."multievolve.env".text = ''
        FILE_VAR=from-environment-file
        HOME=/wrong/home
        MULTIEVOLVE_ROOT=/wrong/root
      '';
      environment.systemPackages = [ pkgs.curl ];
      systemd.tmpfiles.settings."10-sandbox-control"."/srv/sandbox-control".d.mode = "0777";

      services.multievolve-streamlit = {
        enable = true;
        package = testPackage;
        workingDirectory = "/srv/multievolve";
        host = "0.0.0.0";
        port = 18501;
        openFirewall = true;
        environmentFile = "/etc/multievolve.env";
        extraArgs = [
          "--review-value"
          "value with spaces"
        ];
      };
    };

    external = {
      imports = [ module ];

      environment.systemPackages = [ pkgs.curl ];
      systemd.tmpfiles.settings."10-sandbox-control"."/srv/sandbox-control".d.mode = "0777";
      users.groups = {
        accelerator = { };
        deployer = { };
      };
      users.users.deployer = {
        isSystemUser = true;
        group = "deployer";
      };

      services.multievolve-streamlit = {
        enable = true;
        package = testPackage;
        createUser = false;
        user = "deployer";
        group = "deployer";
        extraGroups = [ "accelerator" ];
        port = 18502;
      };
    };
  };

  testScript = ''
    import json

    start_all()
    machine.wait_for_unit("multievolve-streamlit.service")
    machine.wait_for_open_port(18501)

    response = json.loads(machine.succeed("curl --fail --silent http://127.0.0.1:18501"))
    assert response["home"] == "/srv/multievolve"
    assert response["root"] == "/srv/multievolve"
    assert response["file_var"] == "from-environment-file"

    machine.succeed("test $(stat -c %U /srv/multievolve) = multievolve")
    machine.succeed("test $(stat -c %G /srv/multievolve) = multievolve")
    machine.succeed("test $(stat -c %a /srv/multievolve) = 750")
    machine.succeed("test -f /srv/multievolve/service-started")
    machine.succeed("test $(stat -c %a /srv/multievolve/service-started) = 640")
    machine.succeed("test -f /srv/multievolve/sandbox-protected")
    machine.succeed("test $(stat -c %a /srv/sandbox-control) = 777")
    machine.succeed("grep -Fq 'value with spaces' /srv/multievolve/arguments.json")
    machine.fail("test -e /srv/sandbox-control/probe")

    machine.succeed("echo preserved >/srv/multievolve/preserved")
    machine.succeed("systemctl restart multievolve-streamlit.service")
    machine.wait_for_open_port(18501)
    machine.succeed("grep -Fxq preserved /srv/multievolve/preserved")

    external.wait_for_unit("multievolve-streamlit.service")
    external.wait_for_open_port(18502)
    external_response = json.loads(
        external.succeed("curl --fail --silent http://127.0.0.1:18502")
    )
    assert external_response["home"] == "/var/lib/multievolve-streamlit"
    assert external_response["root"] == "/var/lib/multievolve-streamlit"
    assert "accelerator" in external_response["supplementary_groups"]
    external.succeed("curl --fail --silent http://machine:18501 >/dev/null")
    machine.fail("curl --connect-timeout 1 --fail --silent http://external:18502")
    external.succeed("test $(stat -c %U /var/lib/multievolve-streamlit) = deployer")
    external.succeed("test $(stat -c %G /var/lib/multievolve-streamlit) = deployer")
    external.fail("test -e /srv/sandbox-control/probe")
  '';
}
