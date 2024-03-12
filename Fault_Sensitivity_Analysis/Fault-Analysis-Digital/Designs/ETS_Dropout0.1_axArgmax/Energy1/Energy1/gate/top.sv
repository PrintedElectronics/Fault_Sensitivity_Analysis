/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Mon Dec 18 12:01:12 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [31:0] inp;
  output [1:0] out;
  wire   n29, n30, n31, n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42,
         n43, n44, n45, n46;

  AND2X1 U29 ( .A1(n41), .A2(n31), .Y(n45) );
  INVX1 U30 ( .A(n45), .Y(n29) );
  AND2X1 U31 ( .A1(n44), .A2(n43), .Y(n46) );
  INVX1 U32 ( .A(n46), .Y(n30) );
  AND2X1 U33 ( .A1(1'b1), .A2(1'b0), .Y(n42) );
  INVX1 U34 ( .A(n42), .Y(n31) );
  AND2X1 U35 ( .A1(n36), .A2(n35), .Y(n37) );
  INVX1 U36 ( .A(n37), .Y() );
  AND2X1 U37 ( .A1(1'b1), .A2(1'b1), .Y(n38) );
  INVX1 U38 ( .A(n38), .Y() );
  INVX1 U39 ( .A(inp[11]), .Y() );
  NOR2X1 U40 ( .A1(inp[3]), .A2(inp[19]), .Y() );
  NAND2X1 U41 ( .A1(inp[8]), .A2(inp[16]), .Y(n36) );
  NAND2X1 U42 ( .A1(inp[19]), .A2(inp[3]), .Y(n35) );
  INVX1 U43 ( .A(1'b1), .Y(n40) );
  NAND2X1 U44 ( .A1(n40), .A2(inp[11]), .Y(n41) );
  NOR2X1 U45 ( .A1(inp[15]), .A2(inp[24]), .Y(n44) );
  NOR2X1 U46 ( .A1(inp[13]), .A2(inp[11]), .Y(n43) );
  AND2X1 U47 ( .A1(n29), .A2(n30), .Y(out[1]) );
  NOR2X1 U48 ( .A1(n30), .A2(n45), .Y(out[0]) );
endmodule

