/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Mon Dec 18 12:04:02 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [31:0] inp;
  output [1:0] out;
  wire   n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46,
         n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59, n60,
         n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74,
         n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88,
         n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100, n101,
         n102, n103, n104, n105, n106, n107, n108, n109, n110, n111, n112,
         n113, n114, n115, n116, n117, n118, n119, n120, n121, n122, n123;

  NOR2X1 U28 ( .A1(n122), .A2(n121), .Y(out[1]) );
  NAND2X1 U29 ( .A1(n48), .A2(n50), .Y(n33) );
  NAND2X1 U30 ( .A1(inp[11]), .A2(inp[26]), .Y(n34) );
  XNOR2X1 U31 ( .A1(inp[1]), .A2(inp[28]), .Y(n35) );
  NAND2X1 U32 ( .A1(n104), .A2(n47), .Y(n36) );
  NAND2X1 U33 ( .A1(n43), .A2(inp[8]), .Y(n37) );
  NAND2X1 U34 ( .A1(n107), .A2(inp[17]), .Y(n38) );
  NAND2X1 U35 ( .A1(n87), .A2(n62), .Y(n39) );
  NAND2X1 U36 ( .A1(n68), .A2(inp[14]), .Y(n40) );
  AND2X1 U37 ( .A1(n75), .A2(n61), .Y(n41) );
  AND2X1 U38 ( .A1(n63), .A2(n79), .Y(n72) );
  INVX1 U39 ( .A(n72), .Y(n42) );
  AND2X1 U40 ( .A1(n111), .A2(n110), .Y(n112) );
  INVX1 U41 ( .A(n112), .Y(n43) );
  AND2X1 U42 ( .A1(n86), .A2(n85), .Y(n89) );
  INVX1 U43 ( .A(n89), .Y(n44) );
  AND2X1 U44 ( .A1(n99), .A2(n98), .Y(n100) );
  INVX1 U45 ( .A(n100), .Y(n45) );
  AND2X1 U46 ( .A1(n117), .A2(n33), .Y() );
  INVX1 U47 ( .A(1'b1), .Y(n46) );
  AND2X1 U48 ( .A1(n97), .A2(n102), .Y(n103) );
  INVX1 U49 ( .A(n103), .Y(n47) );
  AND2X1 U50 ( .A1(n35), .A2(n114), .Y(n115) );
  INVX1 U51 ( .A(n115), .Y(n48) );
  AND2X1 U52 ( .A1(n36), .A2(n105), .Y(n106) );
  INVX1 U53 ( .A(n106), .Y(n49) );
  AND2X1 U54 ( .A1(n113), .A2(n37), .Y(n116) );
  INVX1 U55 ( .A(n116), .Y(n50) );
  AND2X1 U56 ( .A1(n59), .A2(n79), .Y(n93) );
  INVX1 U57 ( .A(n93), .Y(n51) );
  AND2X1 U58 ( .A1(n74), .A2(inp[19]), .Y(n90) );
  INVX1 U59 ( .A(n90), .Y(n52) );
  AND2X1 U60 ( .A1(n80), .A2(inp[13]), .Y(n92) );
  INVX1 U61 ( .A(n92), .Y(n53) );
  AND2X1 U62 ( .A1(n34), .A2(n61), .Y(n87) );
  INVX1 U63 ( .A(n87), .Y(n54) );
  AND2X1 U64 ( .A1(n91), .A2(n52), .Y(n114) );
  INVX1 U65 ( .A(n114), .Y(n55) );
  AND2X1 U66 ( .A1(n82), .A2(n64), .Y(n96) );
  INVX1 U67 ( .A(n96), .Y(n56) );
  XNOR2X1 U68 ( .A1(n88), .A2(n87), .Y(n97) );
  INVX1 U69 ( .A(n97), .Y(n57) );
  AND2X1 U70 ( .A1(n70), .A2(n69), .Y(n73) );
  INVX1 U71 ( .A(n73), .Y(n58) );
  AND2X1 U72 ( .A1(n78), .A2(n77), .Y(n80) );
  INVX1 U73 ( .A(n80), .Y(n59) );
  AND2X1 U74 ( .A1(n118), .A2(n46), .Y(n123) );
  INVX1 U75 ( .A(n123), .Y(n60) );
  AND2X1 U76 ( .A1(n58), .A2(inp[14]), .Y(n76) );
  INVX1 U77 ( .A(n76), .Y(n61) );
  AND2X1 U78 ( .A1(n40), .A2(n52), .Y(n88) );
  INVX1 U79 ( .A(n88), .Y(n62) );
  AND2X1 U80 ( .A1(n71), .A2(inp[0]), .Y(n95) );
  INVX1 U81 ( .A(n95), .Y(n63) );
  AND2X1 U82 ( .A1(inp[17]), .A2(inp[25]), .Y(n83) );
  INVX1 U83 ( .A(n83), .Y(n64) );
  INVX1 U84 ( .A(n41), .Y(n65) );
  OR2X1 U85 ( .A1(n42), .A2(n64), .Y(n101) );
  INVX1 U86 ( .A(n101), .Y(n66) );
  INVX1 U87 ( .A(n35), .Y(n67) );
  INVX1 U88 ( .A(inp[14]), .Y(n74) );
  INVX1 U89 ( .A(inp[19]), .Y(n68) );
  NAND2X1 U90 ( .A1(n34), .A2(inp[26]), .Y(n70) );
  NAND2X1 U91 ( .A1(n34), .A2(inp[11]), .Y(n69) );
  NAND2X1 U92 ( .A1(n88), .A2(n54), .Y(n86) );
  INVX1 U93 ( .A(inp[24]), .Y(n71) );
  INVX1 U94 ( .A(inp[13]), .Y(n79) );
  NAND2X1 U95 ( .A1(n73), .A2(n74), .Y(n75) );
  NAND2X1 U96 ( .A1(n101), .A2(n65), .Y(n84) );
  NAND2X1 U97 ( .A1(n64), .A2(inp[25]), .Y(n78) );
  NAND2X1 U98 ( .A1(n64), .A2(inp[17]), .Y(n77) );
  NAND2X1 U99 ( .A1(n95), .A2(n51), .Y(n81) );
  NAND2X1 U100 ( .A1(n81), .A2(n53), .Y(n82) );
  NAND2X1 U101 ( .A1(n84), .A2(n56), .Y(n85) );
  NAND2X1 U102 ( .A1(n39), .A2(n44), .Y(n91) );
  NAND2X1 U103 ( .A1(n51), .A2(n53), .Y(n94) );
  XOR2X1 U104 ( .A1(n94), .A2(n63), .Y(n105) );
  NAND2X1 U105 ( .A1(n97), .A2(n66), .Y(n99) );
  NAND2X1 U106 ( .A1(n96), .A2(n57), .Y(n98) );
  NAND2X1 U107 ( .A1(n65), .A2(n45), .Y(n104) );
  NOR2X1 U108 ( .A1(n66), .A2(n65), .Y(n102) );
  NAND2X1 U109 ( .A1(n55), .A2(n49), .Y(n122) );
  INVX1 U110 ( .A(inp[5]), .Y(n107) );
  NAND2X1 U111 ( .A1(n38), .A2(inp[7]), .Y(n117) );
  NOR2X1 U112 ( .A1(inp[8]), .A2(inp[16]), .Y(n108) );
  NAND2X1 U113 ( .A1(n108), .A2(n67), .Y(n113) );
  NOR2X1 U114 ( .A1(inp[17]), .A2(inp[18]), .Y(n109) );
  NAND2X1 U115 ( .A1(n109), .A2(inp[10]), .Y(n111) );
  INVX1 U116 ( .A(inp[16]), .Y(n110) );
  NAND2X1 U117 ( .A1(n33), .A2(inp[18]), .Y(n118) );
  NAND2X1 U118 ( .A1(n35), .A2(n60), .Y(n120) );
  NAND2X1 U119 ( .A1(n120), .A2(inp[8]), .Y(n121) );
  NOR2X1 U120 ( .A1(out[1]), .A2(n123), .Y(out[0]) );
endmodule

