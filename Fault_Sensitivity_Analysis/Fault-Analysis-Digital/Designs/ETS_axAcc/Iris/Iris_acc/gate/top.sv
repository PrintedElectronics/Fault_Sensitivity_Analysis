/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec  7 21:37:57 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [15:0] inp;
  output [1:0] out;
  wire   n26, n27, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37, n38, n39,
         n40, n41, n42, n43, n44, n45, n46, n47, n48, n49, n50, n51, n52, n53,
         n54, n55, n56, n57, n58, n59;

  NAND2X1 U55 ( .A1(n42), .A2(inp[7]), .Y(n26) );
  NAND2X1 U56 ( .A1(n47), .A2(n35), .Y(n27) );
  NAND2X1 U57 ( .A1(n33), .A2(n30), .Y(n28) );
  AND2X1 U58 ( .A1(n39), .A2(1'b0), .Y(n40) );
  INVX1 U59 ( .A(n40), .Y(n29) );
  AND2X1 U60 ( .A1(n55), .A2(n56), .Y(n57) );
  INVX1 U61 ( .A(n57), .Y(n30) );
  AND2X1 U62 ( .A1(n51), .A2(1'b0), .Y(n53) );
  INVX1 U63 ( .A(n53), .Y(n31) );
  AND2X1 U64 ( .A1(n36), .A2(n50), .Y(n37) );
  INVX1 U65 ( .A(n37), .Y(n32) );
  AND2X1 U66 ( .A1(n50), .A2(n49), .Y(n58) );
  INVX1 U67 ( .A(n58), .Y(n33) );
  AND2X1 U68 ( .A1(n28), .A2(inp[11]), .Y(n59) );
  INVX1 U69 ( .A(n59), .Y() );
  AND2X1 U70 ( .A1(1'b1), .A2(n45), .Y(n48) );
  INVX1 U71 ( .A(n48), .Y(n35) );
  NOR2X1 U72 ( .A1(inp[14]), .A2(inp[3]), .Y(n42) );
  NAND2X1 U73 ( .A1(n26), .A2(inp[13]), .Y(n36) );
  INVX1 U74 ( .A(inp[9]), .Y(n50) );
  NAND2X1 U75 ( .A1(n32), .A2(inp[15]), .Y(n39) );
  NAND2X1 U76 ( .A1(inp[9]), .A2(inp[13]), .Y() );
  AND2X1 U77 ( .A1(inp[11]), .A2(n29), .Y(out[1]) );
  NAND2X1 U78 ( .A1(n26), .A2(inp[9]), .Y() );
  NAND2X1 U79 ( .A1(inp[3]), .A2(inp[14]), .Y(n41) );
  NAND2X1 U80 ( .A1(n41), .A2(inp[7]), .Y(n44) );
  INVX1 U81 ( .A(n42), .Y(n43) );
  NAND2X1 U82 ( .A1(n44), .A2(n43), .Y(n45) );
  INVX1 U83 ( .A(inp[11]), .Y(n47) );
  INVX1 U84 ( .A(inp[15]), .Y(n49) );
  INVX1 U85 ( .A(inp[7]), .Y() );
  NOR2X1 U86 ( .A1(inp[5]), .A2(inp[4]), .Y(n51) );
  NOR2X1 U87 ( .A1(inp[0]), .A2(n31), .Y(n54) );
  NOR2X1 U88 ( .A1(inp[13]), .A2(n54), .Y(n56) );
  NAND2X1 U89 ( .A1(inp[15]), .A2(inp[9]), .Y(n55) );
  NAND2X1 U90 ( .A1(n27), .A2(1'b1), .Y(out[0]) );
endmodule

