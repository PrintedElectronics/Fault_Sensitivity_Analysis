/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec  7 21:34:00 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [35:0] inp;
  output [1:0] out;
  wire   n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34,
         n35, n36, n37, n38, n39, n40, n41, n42, n43;
  assign out[1] = 1'b0;

  XOR2X1 U30 ( .A1(inp[32]), .A2(1'b1), .Y(n21) );
  NAND2X1 U31 ( .A1(n35), .A2(1'b1), .Y(n22) );
  AND2X1 U32 ( .A1(1'b1), .A2(inp[32]), .Y(n32) );
  INVX1 U33 ( .A(n32), .Y(n23) );
  AND2X1 U34 ( .A1(n22), .A2(n39), .Y(n40) );
  INVX1 U35 ( .A(n40), .Y(n24) );
  AND2X1 U36 ( .A1(n36), .A2(n37), .Y(n38) );
  INVX1 U37 ( .A(n38), .Y() );
  AND2X1 U38 ( .A1(inp[20]), .A2(inp[9]), .Y(n33) );
  INVX1 U39 ( .A(n33), .Y(n26) );
  AND2X1 U40 ( .A1(n31), .A2(n30), .Y() );
  INVX1 U41 ( .A(1'b0), .Y() );
  OR2X1 U42 ( .A1(n21), .A2(inp[23]), .Y(n35) );
  INVX1 U43 ( .A(n35), .Y(n28) );
  AND2X1 U44 ( .A1(n41), .A2(n24), .Y(n42) );
  INVX1 U45 ( .A(n42), .Y() );
  NOR2X1 U46 ( .A1(inp[30]), .A2(inp[11]), .Y(n43) );
  NAND2X1 U47 ( .A1(n26), .A2(inp[9]), .Y(n31) );
  NAND2X1 U48 ( .A1(n26), .A2(inp[20]), .Y(n30) );
  AND2X1 U49 ( .A1(n26), .A2(n23), .Y(n37) );
  NAND2X1 U50 ( .A1(n37), .A2(n28), .Y(n41) );
  INVX1 U51 ( .A(inp[6]), .Y(n39) );
  NAND2X1 U52 ( .A1(n21), .A2(inp[23]), .Y(n36) );
  NAND2X1 U53 ( .A1(n43), .A2(1'b1), .Y(out[0]) );
endmodule

