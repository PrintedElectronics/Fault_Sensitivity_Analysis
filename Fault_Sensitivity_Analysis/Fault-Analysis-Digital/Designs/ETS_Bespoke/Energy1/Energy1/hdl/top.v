//weights: [[[-6, -1, -6, -2, -5, -5, -4, -4], [37, -18, 31, 44, 75, -1, 6, -1], [-77, 1, -85, -78, 104, 0, -18, 4]], [[4, -19, 95], [-4, 0, 7], [-3, 5, -7]]]
//intercepts: [[-148, -576, 145], [5138, -579, -5864]]
module top (inp, out);
input [31:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    wire signed [11:0] n_0_0_po_0;
    //weight -6 : 8'sb11111010
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11111010;

    wire signed [11:0] n_0_0_po_1;
    //weight -1 : 8'sb11111111
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11111111;

    wire signed [11:0] n_0_0_po_2;
    //weight -6 : 8'sb11111010
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11111010;

    wire signed [11:0] n_0_0_po_3;
    //weight -2 : 8'sb11111110
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11111110;

    wire signed [11:0] n_0_0_po_4;
    //weight -5 : 8'sb11111011
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11111011;

    wire signed [11:0] n_0_0_po_5;
    //weight -5 : 8'sb11111011
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11111011;

    wire signed [11:0] n_0_0_po_6;
    //weight -4 : 8'sb11111100
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb11111100;

    wire signed [11:0] n_0_0_po_7;
    //weight -4 : 8'sb11111100
    assign n_0_0_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb11111100;

    wire signed [19:0] n_0_0_sum;
    assign n_0_0_sum = -148 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_4 + n_0_0_po_5 + n_0_0_po_6 + n_0_0_po_7;
    //relu
    wire [15:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({16{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    wire signed [11:0] n_0_1_po_0;
    //weight 37 : 8'sb00100101
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00100101;

    wire signed [11:0] n_0_1_po_1;
    //weight -18 : 8'sb11101110
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11101110;

    wire signed [11:0] n_0_1_po_2;
    //weight 31 : 8'sb00011111
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00011111;

    wire signed [11:0] n_0_1_po_3;
    //weight 44 : 8'sb00101100
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00101100;

    wire signed [11:0] n_0_1_po_4;
    //weight 75 : 8'sb01001011
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb01001011;

    wire signed [11:0] n_0_1_po_5;
    //weight -1 : 8'sb11111111
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11111111;

    wire signed [11:0] n_0_1_po_6;
    //weight 6 : 8'sb00000110
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb00000110;

    wire signed [11:0] n_0_1_po_7;
    //weight -1 : 8'sb11111111
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb11111111;

    wire signed [19:0] n_0_1_sum;
    assign n_0_1_sum = -576 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_5 + n_0_1_po_6 + n_0_1_po_7;
    //relu
    wire [14:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight -77 : 8'sb10110011
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb10110011;

    wire signed [11:0] n_0_2_po_1;
    //weight 1 : 8'sb00000001
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00000001;

    wire signed [11:0] n_0_2_po_2;
    //weight -85 : 8'sb10101011
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb10101011;

    wire signed [11:0] n_0_2_po_3;
    //weight -78 : 8'sb10110010
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb10110010;

    wire signed [11:0] n_0_2_po_4;
    //weight 104 : 8'sb01101000
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb01101000;

    //weight 0 : skip
    wire signed [11:0] n_0_2_po_6;
    //weight -18 : 8'sb11101110
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb11101110;

    wire signed [11:0] n_0_2_po_7;
    //weight 4 : 8'sb00000100
    assign n_0_2_po_7 = $signed({1'b0, inp[31:28]}) * 8'sb00000100;

    wire signed [19:0] n_0_2_sum;
    assign n_0_2_sum = 145 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_6 + n_0_2_po_7;
    //relu
    wire [13:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({14{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    wire signed [21:0] n_1_0_po_0;
    //weight 4 : 8'sb00000100
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000100;

    wire signed [21:0] n_1_0_po_1;
    //weight -19 : 8'sb11101101
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb11101101;

    wire signed [21:0] n_1_0_po_2;
    //weight 95 : 8'sb01011111
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb01011111;

    wire signed [24:0] n_1_0_sum;
    assign n_1_0_sum = 5138 + n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2;
    //relu
    wire [22:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [21:0] n_1_1_po_0;
    //weight -4 : 8'sb11111100
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb11111100;

    //weight 0 : skip
    wire signed [21:0] n_1_1_po_2;
    //weight 7 : 8'sb00000111
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb00000111;

    wire signed [24:0] n_1_1_sum;
    assign n_1_1_sum = -579 + n_1_1_po_0 + n_1_1_po_2;
    //relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    wire signed [21:0] n_1_2_po_0;
    //weight -3 : 8'sb11111101
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb11111101;

    wire signed [21:0] n_1_2_po_1;
    //weight 5 : 8'sb00000101
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb00000101;

    wire signed [21:0] n_1_2_po_2;
    //weight -7 : 8'sb11111001
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb11111001;

    wire signed [24:0] n_1_2_sum;
    assign n_1_2_sum = -5864 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2;
    //relu
    wire [22:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [23:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [23:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule
