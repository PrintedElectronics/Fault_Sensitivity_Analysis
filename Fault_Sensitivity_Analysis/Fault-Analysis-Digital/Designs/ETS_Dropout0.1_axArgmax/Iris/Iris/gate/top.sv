/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Mon Dec 18 12:05:00 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [15:0] inp;
  output [1:0] out;
  wire   n31, n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44,
         n45, n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58,
         n59, n60, n61, n62, n63, n64;

  NOR2X1 U39 ( .A1(n40), .A2(1'b0), .Y(out[1]) );
  AND2X1 U40 ( .A1(1'b1), .A2(n54), .Y(n56) );
  INVX1 U41 ( .A(n56), .Y(n31) );
  AND2X1 U42 ( .A1(n59), .A2(n58), .Y(n60) );
  INVX1 U43 ( .A(n60), .Y(n32) );
  AND2X1 U44 ( .A1(n51), .A2(1'b1), .Y(n52) );
  INVX1 U45 ( .A(n52), .Y(n33) );
  AND2X1 U46 ( .A1(n63), .A2(n49), .Y(n53) );
  INVX1 U47 ( .A(n53), .Y(n34) );
  AND2X1 U48 ( .A1(n32), .A2(inp[11]), .Y(n61) );
  INVX1 U49 ( .A(n61), .Y() );
  AND2X1 U50 ( .A1(1'b1), .A2(n38), .Y(n63) );
  INVX1 U51 ( .A(n63), .Y(n36) );
  AND2X1 U52 ( .A1(n41), .A2(inp[1]), .Y(n44) );
  INVX1 U53 ( .A(n44), .Y() );
  AND2X1 U54 ( .A1(n47), .A2(n39), .Y(n48) );
  INVX1 U55 ( .A(n48), .Y(n38) );
  AND2X1 U56 ( .A1(n45), .A2(1'b0), .Y(n46) );
  INVX1 U57 ( .A(n46), .Y(n39) );
  AND2X1 U58 ( .A1(n34), .A2(n33), .Y(n62) );
  INVX1 U59 ( .A(n62), .Y(n40) );
  INVX1 U60 ( .A(inp[5]), .Y(n41) );
  NOR2X1 U61 ( .A1(inp[1]), .A2(n41), .Y(n49) );
  INVX1 U62 ( .A(inp[9]), .Y(n51) );
  NOR2X1 U63 ( .A1(1'b0), .A2(n51), .Y(n42) );
  NAND2X1 U64 ( .A1(n42), .A2(inp[13]), .Y(n43) );
  NAND2X1 U65 ( .A1(n43), .A2(inp[7]), .Y(n47) );
  NOR2X1 U66 ( .A1(inp[13]), .A2(inp[9]), .Y(n45) );
  INVX1 U67 ( .A(inp[15]), .Y() );
  NAND2X1 U68 ( .A1(inp[7]), .A2(inp[4]), .Y() );
  NAND2X1 U69 ( .A1(inp[7]), .A2(inp[5]), .Y(n54) );
  NAND2X1 U70 ( .A1(n31), .A2(inp[0]), .Y(n57) );
  NAND2X1 U71 ( .A1(n57), .A2(inp[13]), .Y(n59) );
  NAND2X1 U72 ( .A1(inp[9]), .A2(inp[15]), .Y(n58) );
  NOR2X1 U73 ( .A1(inp[11]), .A2(n36), .Y(n64) );
  NOR2X1 U74 ( .A1(out[1]), .A2(n64), .Y(out[0]) );
endmodule

