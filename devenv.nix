{ pkgs, ... }:

{
  packages = [
    pkgs.git
    pkgs.nodejs
    pkgs.zig
  ];
}
