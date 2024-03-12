/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec 14 00:52:50 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [27:0] inp;
  output [1:0] out;
  wire   n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45,
         n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59;

  NAND2X1 U29 ( .A1(n57), .A2(n38), .Y(n32) );
  NAND2X1 U30 ( .A1(inp[3]), .A2(inp[19]), .Y(n33) );
  NAND2X1 U31 ( .A1(n37), .A2(n41), .Y(n34) );
  AND2X1 U32 ( .A1(n39), .A2(n53), .Y(n54) );
  INVX1 U33 ( .A(n54), .Y(n35) );
  AND2X1 U34 ( .A1(n35), .A2(inp[15]), .Y(n56) );
  INVX1 U35 ( .A(n56), .Y(n36) );
  AND2X1 U36 ( .A1(n48), .A2(n49), .Y(n50) );
  INVX1 U37 ( .A(n50), .Y(n37) );
  AND2X1 U38 ( .A1(n51), .A2(n34), .Y(n58) );
  INVX1 U39 ( .A(n58), .Y(n38) );
  AND2X1 U40 ( .A1(n33), .A2(1'b0), .Y(n52) );
  INVX1 U41 ( .A(n52), .Y(n39) );
  AND2X1 U42 ( .A1(n44), .A2(n45), .Y(n59) );
  INVX1 U43 ( .A(n59), .Y(n40) );
  AND2X1 U44 ( .A1(n47), .A2(n33), .Y() );
  INVX1 U45 ( .A(1'b1), .Y(n41) );
  NOR2X1 U46 ( .A1(inp[11]), .A2(inp[2]), .Y() );
  NOR2X1 U47 ( .A1(1'b1), .A2(inp[27]), .Y(n45) );
  NAND2X1 U48 ( .A1(inp[11]), .A2(inp[2]), .Y(n43) );
  NAND2X1 U49 ( .A1(n43), .A2(inp[23]), .Y(n44) );
  OR2X1 U50 ( .A1(inp[3]), .A2(inp[19]), .Y() );
  NAND2X1 U51 ( .A1(n52), .A2(inp[7]), .Y(n47) );
  INVX1 U52 ( .A(inp[15]), .Y(n49) );
  NAND2X1 U53 ( .A1(n39), .A2(inp[7]), .Y(n48) );
  NAND2X1 U54 ( .A1(inp[17]), .A2(inp[1]), .Y(n51) );
  INVX1 U55 ( .A(inp[7]), .Y(n53) );
  NAND2X1 U56 ( .A1(1'b1), .A2(n36), .Y(n57) );
  AND2X1 U57 ( .A1(n40), .A2(n32), .Y(out[1]) );
  NOR2X1 U58 ( .A1(n59), .A2(n32), .Y(out[0]) );
endmodule

