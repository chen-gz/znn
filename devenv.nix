{ pkgs, ... }:

{
  packages = [
    pkgs.git
    pkgs.nodejs
    pkgs.zig_0_17
  ];
}
