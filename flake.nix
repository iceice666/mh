{
  description = "Rust and MicroQuickJS development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    mquickjs = {
      url = "github:bellard/mquickjs";
      flake = false;
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      mquickjs,
      ...
    }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          mquickjs = pkgs.stdenv.mkDerivation {
            pname = "mquickjs";
            version = "unstable";
            src = mquickjs;

            nativeBuildInputs = [ pkgs.gnumake ];

            buildPhase = ''
              runHook preBuild
              make -j$NIX_BUILD_CORES CC="$CC" HOST_CC="$CC"
              "$AR" rcs libmquickjs.a mquickjs.o dtoa.o libm.o cutils.o
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall

              mkdir -p "$out/bin" "$out/include" "$out/lib/pkgconfig" "$out/share/mquickjs"
              install -m755 mqjs example "$out/bin/"
              install -m644 libmquickjs.a "$out/lib/"
              install -m644 mquickjs.h cutils.h dtoa.h libm.h list.h "$out/include/"
              cp -R tests "$out/share/mquickjs/"

              cat > "$out/lib/pkgconfig/mquickjs.pc" <<EOF
              prefix=$out
              exec_prefix=\''${prefix}
              libdir=\''${prefix}/lib
              includedir=\''${prefix}/include

              Name: mquickjs
              Description: MicroQuickJS embedded JavaScript engine
              Version: unstable
              Libs: -L\''${libdir} -lmquickjs
              Libs.private: -lm
              Cflags: -I\''${includedir}
              EOF

              runHook postInstall
            '';

            meta = {
              description = "Small embeddable JavaScript engine for microcontrollers";
              homepage = "https://github.com/bellard/mquickjs";
              license = pkgs.lib.licenses.mit;
              platforms = systems;
              mainProgram = "mqjs";
            };
          };

          default = self.packages.${system}.mquickjs;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          mquickjsPackage = self.packages.${system}.mquickjs;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              clippy
              pkg-config
              rust-analyzer
              rustc
              rustfmt
              mquickjsPackage
            ];

            MQUICKJS_ROOT = "${mquickjsPackage}";
            MQUICKJS_INCLUDE_DIR = "${mquickjsPackage}/include";
            MQUICKJS_LIB_DIR = "${mquickjsPackage}/lib";
            RUST_BACKTRACE = "1";
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
