/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec 14 00:48:42 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [31:0] inp;
  output [1:0] out;
  wire   n51, n52, n53, n54, n55, n56, n57, n58, n59, n60, n61, n62, n63, n64,
         n65, n66, n67, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77, n78,
         n79, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91, n92,
         n93, n94, n95, n96, n97, n98, n99, n100, n101, n102, n103, n104, n105,
         n106, n107, n108, n109;

  NAND2X1 U50 ( .A1(inp[11]), .A2(inp[18]), .Y(n51) );
  AND2X1 U51 ( .A1(n83), .A2(n55), .Y(n93) );
  INVX1 U52 ( .A(n93), .Y(n52) );
  NAND2X1 U53 ( .A1(n85), .A2(inp[25]), .Y(n53) );
  NAND2X1 U54 ( .A1(n60), .A2(n68), .Y(n54) );
  NOR2X1 U55 ( .A1(inp[8]), .A2(n62), .Y(n55) );
  NOR2X1 U56 ( .A1(n84), .A2(inp[15]), .Y(n56) );
  AND2X1 U57 ( .A1(n85), .A2(inp[16]), .Y(n82) );
  INVX1 U58 ( .A(n82), .Y() );
  NAND2X1 U59 ( .A1(n91), .A2(n65), .Y(n58) );
  NAND2X1 U60 ( .A1(n79), .A2(inp[25]), .Y(n59) );
  NAND2X1 U61 ( .A1(n105), .A2(inp[15]), .Y(n60) );
  AND2X1 U62 ( .A1(n74), .A2(inp[25]), .Y(n86) );
  INVX1 U63 ( .A(n86), .Y(n61) );
  AND2X1 U64 ( .A1(n84), .A2(inp[15]), .Y(n75) );
  INVX1 U65 ( .A(n75), .Y(n62) );
  AND2X1 U66 ( .A1(n76), .A2(n98), .Y(n80) );
  INVX1 U67 ( .A(n80), .Y(n63) );
  AND2X1 U68 ( .A1(n98), .A2(n97), .Y(n99) );
  INVX1 U69 ( .A(n99), .Y(n64) );
  AND2X1 U70 ( .A1(n89), .A2(n51), .Y(n90) );
  INVX1 U71 ( .A(n90), .Y(n65) );
  AND2X1 U72 ( .A1(n58), .A2(n56), .Y(n92) );
  INVX1 U73 ( .A(n92), .Y(n66) );
  AND2X1 U74 ( .A1(n59), .A2(n63), .Y(n81) );
  INVX1 U75 ( .A(n81), .Y(n67) );
  AND2X1 U76 ( .A1(n64), .A2(n71), .Y(n106) );
  INVX1 U77 ( .A(n106), .Y(n68) );
  AND2X1 U78 ( .A1(n101), .A2(inp[25]), .Y(n104) );
  INVX1 U79 ( .A(n104), .Y(n69) );
  AND2X1 U80 ( .A1(n54), .A2(1'b0), .Y(n108) );
  INVX1 U81 ( .A(n108), .Y(n70) );
  AND2X1 U82 ( .A1(n96), .A2(n95), .Y(n100) );
  INVX1 U83 ( .A(n100), .Y(n71) );
  AND2X1 U84 ( .A1(n97), .A2(1'b0), .Y(n94) );
  INVX1 U85 ( .A(n94), .Y(n72) );
  AND2X1 U86 ( .A1(n78), .A2(inp[21]), .Y(n101) );
  INVX1 U87 ( .A(n101), .Y(n73) );
  AND2X1 U88 ( .A1(n51), .A2(1'b0), .Y(n85) );
  INVX1 U89 ( .A(n85), .Y(n74) );
  INVX1 U90 ( .A(inp[19]), .Y(n84) );
  OR2X1 U91 ( .A1(inp[11]), .A2(inp[18]), .Y() );
  INVX1 U92 ( .A(inp[13]), .Y(n78) );
  NOR2X1 U93 ( .A1(n78), .A2(inp[21]), .Y(n77) );
  NAND2X1 U94 ( .A1(n77), .A2(inp[17]), .Y(n76) );
  INVX1 U95 ( .A(inp[25]), .Y(n98) );
  INVX1 U96 ( .A(inp[17]), .Y(n97) );
  INVX1 U97 ( .A(n77), .Y() );
  NAND2X1 U98 ( .A1(n94), .A2(n73), .Y(n79) );
  NOR2X1 U99 ( .A1(1'b1), .A2(n67), .Y(n83) );
  NOR2X1 U100 ( .A1(inp[16]), .A2(n61), .Y() );
  NAND2X1 U101 ( .A1(1'b1), .A2(inp[17]), .Y(n91) );
  NAND2X1 U102 ( .A1(n53), .A2(inp[17]), .Y(n88) );
  NAND2X1 U103 ( .A1(n88), .A2(n72), .Y(n89) );
  NAND2X1 U104 ( .A1(n52), .A2(n66), .Y(out[0]) );
  NAND2X1 U105 ( .A1(n72), .A2(inp[25]), .Y(n96) );
  OR2X1 U106 ( .A1(n56), .A2(n55), .Y(n95) );
  NAND2X1 U107 ( .A1(1'b0), .A2(inp[17]), .Y(n103) );
  NAND2X1 U108 ( .A1(n103), .A2(n69), .Y(n105) );
  NAND2X1 U109 ( .A1(n51), .A2(n70), .Y(n109) );
  AND2X1 U110 ( .A1(inp[16]), .A2(n109), .Y(out[1]) );
endmodule

