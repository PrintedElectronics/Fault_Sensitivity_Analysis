/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec 14 00:49:41 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [15:0] inp;
  output [1:0] out;
  wire   n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n47, n48,
         n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59, n60, n61, n62,
         n63, n64, n65, n66, n67, n68, n69, n70;

  AND2X1 U40 ( .A1(n58), .A2(n36), .Y(n68) );
  INVX1 U41 ( .A(n68), .Y(n35) );
  NOR2X1 U42 ( .A1(inp[14]), .A2(n46), .Y(n36) );
  NAND2X1 U43 ( .A1(n35), .A2(n43), .Y(n37) );
  NAND2X1 U44 ( .A1(n60), .A2(inp[3]), .Y(n38) );
  AND2X1 U45 ( .A1(n56), .A2(n55), .Y(n39) );
  AND2X1 U46 ( .A1(n48), .A2(n47), .Y(n49) );
  INVX1 U47 ( .A(n49), .Y(n40) );
  AND2X1 U48 ( .A1(n56), .A2(n57), .Y(n51) );
  INVX1 U49 ( .A(n51), .Y(n41) );
  AND2X1 U50 ( .A1(n61), .A2(n38), .Y(n62) );
  INVX1 U51 ( .A(n62), .Y(n42) );
  AND2X1 U52 ( .A1(n65), .A2(n66), .Y(n67) );
  INVX1 U53 ( .A(n67), .Y(n43) );
  AND2X1 U54 ( .A1(n53), .A2(n52), .Y(n54) );
  INVX1 U55 ( .A(n54), .Y(n44) );
  AND2X1 U56 ( .A1(n50), .A2(n40), .Y(n55) );
  INVX1 U57 ( .A(n55), .Y(n45) );
  AND2X1 U58 ( .A1(n63), .A2(n57), .Y(n59) );
  INVX1 U59 ( .A(n59), .Y(n46) );
  INVX1 U60 ( .A(inp[13]), .Y(n50) );
  NOR2X1 U61 ( .A1(inp[7]), .A2(inp[0]), .Y(n48) );
  NOR2X1 U62 ( .A1(inp[5]), .A2(inp[4]), .Y(n47) );
  INVX1 U63 ( .A(inp[15]), .Y(n56) );
  INVX1 U64 ( .A(inp[9]), .Y(n57) );
  NAND2X1 U65 ( .A1(n41), .A2(n45), .Y(n53) );
  NAND2X1 U66 ( .A1(inp[9]), .A2(inp[15]), .Y(n52) );
  AND2X1 U67 ( .A1(n44), .A2(inp[11]), .Y(n69) );
  NOR2X1 U68 ( .A1(n39), .A2(inp[0]), .Y(n58) );
  INVX1 U69 ( .A(inp[7]), .Y(n60) );
  NOR2X1 U70 ( .A1(inp[3]), .A2(n60), .Y(n63) );
  NAND2X1 U71 ( .A1(n46), .A2(inp[14]), .Y(n61) );
  NOR2X1 U72 ( .A1(inp[11]), .A2(n42), .Y(n66) );
  INVX1 U73 ( .A(n63), .Y(n64) );
  NAND2X1 U74 ( .A1(n64), .A2(inp[9]), .Y(n65) );
  NOR2X1 U75 ( .A1(n69), .A2(n37), .Y(out[0]) );
  INVX1 U76 ( .A(n69), .Y(n70) );
  NOR2X1 U77 ( .A1(n36), .A2(n70), .Y(out[1]) );
endmodule

