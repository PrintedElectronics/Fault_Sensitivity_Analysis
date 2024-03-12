//weights: [[[-29, 48, -62, -57], [0, -7, 5, 18], [-15, -1, 108, 31]], [[71, 30, -105], [-69, -13, -75], [-55, -42, 75]]]
//intercepts: [[639, -698, -1112], [-11255, 17487, -17315]]
module top (inp, out);
input [15:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    wire signed [11:0] n_0_0_po_0;
    //weight -29 : 8'sb11100011
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11100011;

    wire signed [11:0] n_0_0_po_1;
    //weight 48 : 8'sb00110000
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00110000;

    wire signed [11:0] n_0_0_po_2;
    //weight -62 : 8'sb11000010
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11000010;

    wire signed [11:0] n_0_0_po_3;
    //weight -57 : 8'sb11000111
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11000111;

    wire signed [15:0] n_0_0_sum;
    assign n_0_0_sum = 639 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3;
    //relu
    wire [13:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({14{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight 0 : skip
    wire signed [11:0] n_0_1_po_1;
    //weight -7 : 8'sb11111001
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11111001;

    wire signed [11:0] n_0_1_po_2;
    //weight 5 : 8'sb00000101
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00000101;

    wire signed [11:0] n_0_1_po_3;
    //weight 18 : 8'sb00010010
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00010010;

    wire signed [15:0] n_0_1_sum;
    assign n_0_1_sum = -698 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3;
    //relu
    wire [12:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({13{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight -15 : 8'sb11110001
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11110001;

    wire signed [11:0] n_0_2_po_1;
    //weight -1 : 8'sb11111111
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11111111;

    wire signed [11:0] n_0_2_po_2;
    //weight 108 : 8'sb01101100
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb01101100;

    wire signed [11:0] n_0_2_po_3;
    //weight 31 : 8'sb00011111
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00011111;

    wire signed [15:0] n_0_2_sum;
    assign n_0_2_sum = -1112 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3;
    //relu
    wire [13:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({14{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    wire signed [20:0] n_1_0_po_0;
    //weight 71 : 8'sb01000111
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb01000111;

    wire signed [20:0] n_1_0_po_1;
    //weight 30 : 8'sb00011110
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb00011110;

    wire signed [20:0] n_1_0_po_2;
    //weight -105 : 8'sb10010111
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb10010111;

    wire signed [23:0] n_1_0_sum;
    assign n_1_0_sum = -11255 + n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2;
    //relu
    wire [21:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [20:0] n_1_1_po_0;
    //weight -69 : 8'sb10111011
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb10111011;

    wire signed [20:0] n_1_1_po_1;
    //weight -13 : 8'sb11110011
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb11110011;

    wire signed [20:0] n_1_1_po_2;
    //weight -75 : 8'sb10110101
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb10110101;

    wire signed [23:0] n_1_1_sum;
    assign n_1_1_sum = 17487 + n_1_1_po_0 + n_1_1_po_1 + n_1_1_po_2;
    //relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    wire signed [20:0] n_1_2_po_0;
    //weight -55 : 8'sb11001001
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb11001001;

    wire signed [20:0] n_1_2_po_1;
    //weight -42 : 8'sb11010110
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb11010110;

    wire signed [20:0] n_1_2_po_2;
    //weight 75 : 8'sb01001011
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb01001011;

    wire signed [23:0] n_1_2_sum;
    assign n_1_2_sum = -17315 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    //relu
    wire [22:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [22:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [22:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule
