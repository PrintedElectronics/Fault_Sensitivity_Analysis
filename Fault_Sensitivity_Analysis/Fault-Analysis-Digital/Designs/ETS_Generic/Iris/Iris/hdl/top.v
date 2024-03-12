//generic design, we don't actually know the weights and intercepts, they are given to the network through the testbench.
//weights: [[[-29, 48, -62, -57], [0, -7, 5, 18], [-15, -1, 108, 31]], [[71, 30, -105], [-69, -13, -75], [-55, -42, 75]]]
//intercepts: [[639, -698, -1112], [-11255, 17487, -17315]]
module top (inp, weights, biases, out);
input [15:0] inp;
input [167:0] weights;
input [83:0] biases;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight abs(-29) : 8'b00011101
    wire signed [12-1:0] n_0_0_po_0;
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[8-1:0]);

    //weight 48 : 8'b00110000
    wire signed [12-1:0] n_0_0_po_1;
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[16-1:8]);

    //weight abs(-62) : 8'b00111110
    wire signed [12-1:0] n_0_0_po_2;
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[24-1:16]);

    //weight abs(-57) : 8'b00111001
    wire signed [12-1:0] n_0_0_po_3;
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[32-1:24]);

    //accumulate products
    wire signed [15:0] n_0_0_sum;
    assign n_0_0_sum = $signed(biases[12-1:0]) + $signed(n_0_0_po_0) + $signed(n_0_0_po_1) + $signed(n_0_0_po_2) + $signed(n_0_0_po_3);
    //relu
    wire [14:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_1_po_0;
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[40-1:32]);

    //weight abs(-7) : 8'b00000111
    wire signed [12-1:0] n_0_1_po_1;
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[48-1:40]);

    //weight 5 : 8'b00000101
    wire signed [12-1:0] n_0_1_po_2;
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[56-1:48]);

    //weight 18 : 8'b00010010
    wire signed [12-1:0] n_0_1_po_3;
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[64-1:56]);

    //accumulate products
    wire signed [15:0] n_0_1_sum;
    assign n_0_1_sum = $signed(biases[24-1:12]) + $signed(n_0_1_po_0) + $signed(n_0_1_po_1) + $signed(n_0_1_po_2) + $signed(n_0_1_po_3);
    //relu
    wire [14:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    //weight abs(-15) : 8'b00001111
    wire signed [12-1:0] n_0_2_po_0;
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[72-1:64]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_2_po_1;
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[80-1:72]);

    //weight 108 : 8'b01101100
    wire signed [12-1:0] n_0_2_po_2;
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[88-1:80]);

    //weight 31 : 8'b00011111
    wire signed [12-1:0] n_0_2_po_3;
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[96-1:88]);

    //accumulate products
    wire signed [15:0] n_0_2_sum;
    assign n_0_2_sum = $signed(biases[36-1:24]) + $signed(n_0_2_po_0) + $signed(n_0_2_po_1) + $signed(n_0_2_po_2) + $signed(n_0_2_po_3);
    //relu
    wire [14:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    //weight 71 : 8'b01000111
    wire signed [21-1:0] n_1_0_po_0;
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[104-1:96]);

    //weight 30 : 8'b00011110
    wire signed [21-1:0] n_1_0_po_1;
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[112-1:104]);

    //weight abs(-105) : 8'b01101001
    wire signed [21-1:0] n_1_0_po_2;
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[120-1:112]);

    //accumulate products
    wire signed [23:0] n_1_0_sum;
    assign n_1_0_sum = $signed(biases[52-1:36]) + $signed(n_1_0_po_0) + $signed(n_1_0_po_1) + $signed(n_1_0_po_2);
    //relu
    wire [22:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    //weight abs(-69) : 8'b01000101
    wire signed [21-1:0] n_1_1_po_0;
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[128-1:120]);

    //weight abs(-13) : 8'b00001101
    wire signed [21-1:0] n_1_1_po_1;
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[136-1:128]);

    //weight abs(-75) : 8'b01001011
    wire signed [21-1:0] n_1_1_po_2;
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[144-1:136]);

    //accumulate products
    wire signed [23:0] n_1_1_sum;
    assign n_1_1_sum = $signed(biases[68-1:52]) + $signed(n_1_1_po_0) + $signed(n_1_1_po_1) + $signed(n_1_1_po_2);
    //relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight abs(-55) : 8'b00110111
    wire signed [21-1:0] n_1_2_po_0;
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[152-1:144]);

    //weight abs(-42) : 8'b00101010
    wire signed [21-1:0] n_1_2_po_1;
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[160-1:152]);

    //weight 75 : 8'b01001011
    wire signed [21-1:0] n_1_2_po_2;
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[168-1:160]);

    //accumulate products
    wire signed [23:0] n_1_2_sum;
    assign n_1_2_sum = $signed(biases[84-1:68]) + $signed(n_1_2_po_0) + $signed(n_1_2_po_1) + $signed(n_1_2_po_2);
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
