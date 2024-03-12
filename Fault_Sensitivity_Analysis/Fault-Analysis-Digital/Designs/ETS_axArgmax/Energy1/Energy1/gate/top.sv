/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec 14 00:47:45 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [31:0] inp;
  output [1:0] out;
  wire   n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59,
         n60, n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73,
         n74, n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87,
         n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100,
         n101, n102, n103, n104, n105, n106, n107, n108, n109, n110, n111;

  NOR2X1 U53 ( .A1(n110), .A2(n51), .Y(out[1]) );
  NAND2X1 U54 ( .A1(n66), .A2(inp[17]), .Y(n46) );
  NAND2X1 U55 ( .A1(n54), .A2(inp[1]), .Y(n47) );
  NAND2X1 U56 ( .A1(n77), .A2(inp[14]), .Y(n48) );
  NAND2X1 U57 ( .A1(n85), .A2(n61), .Y(n49) );
  AND2X1 U58 ( .A1(n91), .A2(1'b1), .Y(n92) );
  INVX1 U59 ( .A(n92), .Y(n50) );
  AND2X1 U60 ( .A1(n108), .A2(n107), .Y(n109) );
  INVX1 U61 ( .A(n109), .Y(n51) );
  AND2X1 U62 ( .A1(n78), .A2(n48), .Y(n79) );
  INVX1 U63 ( .A(n79), .Y(n52) );
  AND2X1 U64 ( .A1(n96), .A2(n95), .Y(n97) );
  INVX1 U65 ( .A(n97), .Y() );
  AND2X1 U66 ( .A1(n94), .A2(n64), .Y(n67) );
  INVX1 U67 ( .A(n67), .Y(n54) );
  AND2X1 U68 ( .A1(n81), .A2(n80), .Y(n82) );
  INVX1 U69 ( .A(n82), .Y(n55) );
  OR2X1 U70 ( .A1(n66), .A2(inp[17]), .Y(n70) );
  INVX1 U71 ( .A(n70), .Y(n56) );
  XNOR2X1 U72 ( .A1(inp[14]), .A2(inp[2]), .Y(n74) );
  INVX1 U73 ( .A(n74), .Y(n57) );
  XNOR2X1 U74 ( .A1(inp[1]), .A2(inp[17]), .Y(n88) );
  INVX1 U75 ( .A(n88), .Y(n58) );
  OR2X1 U76 ( .A1(n93), .A2(n50), .Y(n106) );
  INVX1 U77 ( .A(n106), .Y(n59) );
  AND2X1 U78 ( .A1(n69), .A2(n47), .Y(n76) );
  INVX1 U79 ( .A(n76), .Y(n60) );
  AND2X1 U80 ( .A1(inp[16]), .A2(inp[8]), .Y(n89) );
  INVX1 U81 ( .A(n89), .Y(n61) );
  AND2X1 U82 ( .A1(inp[11]), .A2(inp[3]), .Y(n102) );
  INVX1 U83 ( .A(n102), .Y(n62) );
  AND2X1 U84 ( .A1(n73), .A2(n72), .Y(n75) );
  INVX1 U85 ( .A(n75), .Y() );
  AND2X1 U86 ( .A1(n46), .A2(n70), .Y(n68) );
  INVX1 U87 ( .A(n68), .Y(n64) );
  AND2X1 U88 ( .A1(n104), .A2(n87), .Y(n111) );
  INVX1 U89 ( .A(n111), .Y(n65) );
  INVX1 U90 ( .A(inp[13]), .Y(n66) );
  INVX1 U91 ( .A(inp[24]), .Y(n94) );
  NAND2X1 U92 ( .A1(n68), .A2(inp[24]), .Y(n69) );
  NAND2X1 U93 ( .A1(n70), .A2(n60), .Y(n73) );
  NAND2X1 U94 ( .A1(inp[1]), .A2(inp[24]), .Y(n71) );
  NAND2X1 U95 ( .A1(n71), .A2(n56), .Y(n72) );
  NOR2X1 U96 ( .A1(1'b1), .A2(n57), .Y(n83) );
  NAND2X1 U97 ( .A1(1'b1), .A2(n57), .Y(n81) );
  NAND2X1 U98 ( .A1(n60), .A2(inp[2]), .Y(n78) );
  OR2X1 U99 ( .A1(inp[2]), .A2(n60), .Y(n77) );
  NOR2X1 U100 ( .A1(inp[11]), .A2(n52), .Y(n80) );
  NOR2X1 U101 ( .A1(n83), .A2(n55), .Y(n87) );
  NOR2X1 U102 ( .A1(inp[16]), .A2(inp[8]), .Y(n86) );
  NOR2X1 U103 ( .A1(inp[11]), .A2(inp[3]), .Y(n100) );
  INVX1 U104 ( .A(inp[19]), .Y() );
  NAND2X1 U105 ( .A1(n100), .A2(1'b1), .Y(n85) );
  NOR2X1 U106 ( .A1(n86), .A2(n49), .Y(n104) );
  NOR2X1 U107 ( .A1(n68), .A2(inp[1]), .Y(n98) );
  NOR2X1 U108 ( .A1(n61), .A2(n58), .Y(n93) );
  NAND2X1 U109 ( .A1(n102), .A2(inp[19]), .Y(n91) );
  NAND2X1 U110 ( .A1(n61), .A2(n58), .Y() );
  NAND2X1 U111 ( .A1(n94), .A2(n59), .Y(n96) );
  NAND2X1 U112 ( .A1(n68), .A2(inp[1]), .Y(n95) );
  NOR2X1 U113 ( .A1(n98), .A2(1'b0), .Y(n99) );
  NOR2X1 U114 ( .A1(n65), .A2(n99), .Y(n110) );
  INVX1 U115 ( .A(n100), .Y(n101) );
  NAND2X1 U116 ( .A1(n101), .A2(inp[19]), .Y(n103) );
  NAND2X1 U117 ( .A1(n103), .A2(n62), .Y(n108) );
  INVX1 U118 ( .A(n104), .Y(n105) );
  NAND2X1 U119 ( .A1(n59), .A2(n105), .Y(n107) );
  NOR2X1 U120 ( .A1(out[1]), .A2(n65), .Y(out[0]) );
endmodule

