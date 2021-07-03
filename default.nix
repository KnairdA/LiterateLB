{ pkgs ? import <nixpkgs> {
  overlays = [
    (import (builtins.fetchTarball {
      url = https://github.com/nix-community/emacs-overlay/archive/0bb3c36bb8cddd92b788d8ce474c39475148d5e2.tar.gz;
    }))
  ];
}, ... }:

let
  cuda-samples-common-headers = pkgs.stdenv.mkDerivation rec {
    name = "cuda-samples-common-headers";
    version = "11.1";

    src = pkgs.fetchFromGitHub {
      owner = "NVIDIA";
      repo = "cuda-samples";
      rev = "v${version}";
      sha256 = "1kjixk50i8y1bkiwbdn5lkv342crvkmbvy1xl5j3lsa1ica21kwh";
    };

    phases = [ "installPhase" ];

    installPhase = ''
      mkdir -p $out/include/cuda-samples/Common
      cp -r $src/Common/* $out/include/cuda-samples/Common
    '';
  };
    
  imgui-sfml = pkgs.stdenv.mkDerivation rec {
    name = "imgui-sfml";
    version = "2.1";

    src = pkgs.fetchFromGitHub {
      owner = "eliasdaler";
      repo = "imgui-sfml";
      rev = "v${version}";
      sha256 = "1g8gqly156miv12ajapnhmxfcv9i3fqhdmdy45gmdw47kh8ly5zj";
    }; 

    buildInputs = with pkgs; [
      cmake
    ];

    propagatedBuildInputs = with pkgs; [
      libGL
      sfml
      xorg.libX11
    ];

    cmakeFlags = let
      imgui_src = pkgs.fetchFromGitHub {
        owner = "ocornut";
        repo = "imgui";
        rev = "v1.68";
        sha256 = "0a7b4fljybvpls84rqzsb2p4r89ic2g6w2m9h0209xlhm4k0x7qr";
      };
    in [
      "-DIMGUI_DIR=${imgui_src}"
      "-DBUILD_SHARED_LIBS=ON"
    ];
  };

in pkgs.stdenv.mkDerivation rec {
  name = "LiterateLB";

  src = ./.;

  nativeBuildInputs = with pkgs; [
    cmake
    addOpenGLRunpath
  ];

  buildInputs = let
    local-python = pkgs.python3.withPackages (python-packages: with python-packages; [
      sympy
      numpy
      Mako
      scipy
      matplotlib
    ]);

  in with pkgs; [
    local-python
    cudatoolkit_11
    cuda-samples-common-headers
    linuxPackages.nvidia_x11 
    libGL
    sfml
    imgui-sfml
    glm
  ];

  phases = [ "buildPhase" "installPhase" ];

  buildPhase = let
    tangle-el = pkgs.writeTextFile {
      name = "tangle.el";
      text = ''
        (toggle-debug-on-error)
        (require 'org)
        (org-babel-do-load-languages 'org-babel-load-languages '((python  . t)))
        (setq org-confirm-babel-evaluate nil)
        (setq org-babel-confirm-evaluate-answer-no t)
        (find-file "lbm.org")
        (setq org-babel-default-header-args
          (cons '(:result-params . ("none"))
                (assq-delete-all :result-params org-babel-default-header-args)))
        (org-babel-execute-buffer)
        (setq org-babel-default-header-args
          (cons '(:result-params . ("output"))
                (assq-delete-all :result-params org-babel-default-header-args)))
        (org-babel-tangle)
      '';
    };

  in ''
    cp $src/lbm.org .
    mkdir -p tangle/asset
    cp -r $src/tangle/asset/noise tangle/asset
    cp -r $src/tangle/asset/palette tangle/asset
    ${pkgs.emacsUnstable-nox}/bin/emacs --no-init-file --script ${tangle-el}
    cp $src/CMakeLists.txt .
    cp $src/assets.cmake .
    mkdir build
    pushd build
    cmake ..
    make
    popd
  '';

  installPhase = ''
    mkdir -p $out/bin
    for program in tangle/*.cu; do
        cp build/$(basename $program .cu) $out/bin/$(basename $program .cu)
        addOpenGLRunpath $out/bin/$(basename $program .cu)
    done

    mkdir $out/include
    cp -r tangle/LLBM $out/include/
  '';

  env = pkgs.buildEnv {
    name = name;
    paths = buildInputs;
  };

  shellHook = ''
    export NIX_SHELL_NAME="${name}"
    export CUDA_PATH="${pkgs.cudatoolkit_11}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib
  '';
}
