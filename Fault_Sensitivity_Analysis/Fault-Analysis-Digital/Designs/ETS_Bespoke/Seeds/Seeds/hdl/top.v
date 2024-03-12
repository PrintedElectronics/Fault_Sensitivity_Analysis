//weights: [[[-6, -7, -9, -5, -10, -5, -7], [12, 6, 21, 109, -36, -13, -95], [58, 64, -23, 46, 48, -36, 49]], [[0, 110, -19], [2, -105, 23], [0, -76, -43]]]
//intercepts: [[-176, 73, -908], [-4678, -3411, 3343]]
module top (inp, out);
input [27:0] inp;
output [1:0] out;

// layer: 0 - neuron: 0
    wire signed [11:0] n_0_0_po_0;
    //weight -6 : 8'sb11111010
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb11111010;

    wire signed [11:0] n_0_0_po_1;
    //weight -7 : 8'sb11111001
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb11111001;

    wire signed [11:0] n_0_0_po_2;
    //weight -9 : 8'sb11110111
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11110111;

    wire signed [11:0] n_0_0_po_3;
    //weight -5 : 8'sb11111011
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb11111011;

    wire signed [11:0] n_0_0_po_4;
    //weight -10 : 8'sb11110110
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11110110;

    wire signed [11:0] n_0_0_po_5;
    //weight -5 : 8'sb11111011
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11111011;

    wire signed [11:0] n_0_0_po_6;
    //weight -7 : 8'sb11111001
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb11111001;

    wire signed [18:0] n_0_0_sum;
    assign n_0_0_sum = -176 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_4 + n_0_0_po_5 + n_0_0_po_6;
    //relu
    wire [14:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    wire signed [11:0] n_0_1_po_0;
    //weight 12 : 8'sb00001100
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00001100;

    wire signed [11:0] n_0_1_po_1;
    //weight 6 : 8'sb00000110
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb00000110;

    wire signed [11:0] n_0_1_po_2;
    //weight 21 : 8'sb00010101
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb00010101;

    wire signed [11:0] n_0_1_po_3;
    //weight 109 : 8'sb01101101
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb01101101;

    wire signed [11:0] n_0_1_po_4;
    //weight -36 : 8'sb11011100
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb11011100;

    wire signed [11:0] n_0_1_po_5;
    //weight -13 : 8'sb11110011
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11110011;

    wire signed [11:0] n_0_1_po_6;
    //weight -95 : 8'sb10100001
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb10100001;

    wire signed [18:0] n_0_1_sum;
    assign n_0_1_sum = 73 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4 + n_0_1_po_5 + n_0_1_po_6;
    //relu
    wire [14:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    wire signed [11:0] n_0_2_po_0;
    //weight 58 : 8'sb00111010
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * 8'sb00111010;

    wire signed [11:0] n_0_2_po_1;
    //weight 64 : 8'sb01000000
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * 8'sb01000000;

    wire signed [11:0] n_0_2_po_2;
    //weight -23 : 8'sb11101001
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * 8'sb11101001;

    wire signed [11:0] n_0_2_po_3;
    //weight 46 : 8'sb00101110
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * 8'sb00101110;

    wire signed [11:0] n_0_2_po_4;
    //weight 48 : 8'sb00110000
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * 8'sb00110000;

    wire signed [11:0] n_0_2_po_5;
    //weight -36 : 8'sb11011100
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * 8'sb11011100;

    wire signed [11:0] n_0_2_po_6;
    //weight 49 : 8'sb00110001
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * 8'sb00110001;

    wire signed [18:0] n_0_2_sum;
    assign n_0_2_sum = -908 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_4 + n_0_2_po_5 + n_0_2_po_6;
    //relu
    wire [14:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    //weight 0 : skip
    wire signed [21:0] n_1_0_po_1;
    //weight 110 : 8'sb01101110
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb01101110;

    wire signed [21:0] n_1_0_po_2;
    //weight -19 : 8'sb11101101
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb11101101;

    wire signed [24:0] n_1_0_sum;
    assign n_1_0_sum = -4678 + n_1_0_po_1 + n_1_0_po_2;
    //relu
    wire [22:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    wire signed [21:0] n_1_1_po_0;
    //weight 2 : 8'sb00000010
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000010;

    wire signed [21:0] n_1_1_po_1;
    //weight -105 : 8'sb10010111
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb10010111;

    wire signed [21:0] n_1_1_po_2;
    //weight 23 : 8'sb00010111
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb00010111;

    wire signed [24:0] n_1_1_sum;
    assign n_1_1_sum = -3411 + n_1_1_po_0 + n_1_1_po_1 + n_1_1_po_2;
    //relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight 0 : skip
    wire signed [21:0] n_1_2_po_1;
    //weight -76 : 8'sb10110100
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb10110100;

    wire signed [21:0] n_1_2_po_2;
    //weight -43 : 8'sb11010101
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb11010101;

    wire signed [24:0] n_1_2_sum;
    assign n_1_2_sum = 3343 + n_1_2_po_1 + n_1_2_po_2;
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
