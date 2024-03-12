`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=2;
    parameter NUM_A=21;
    parameter WIDTH_A=4;
    parameter NUM_W=72;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=12;
    parameter NUM_B1=3;
    parameter WIDTH_B1=13;

localparam period = 250000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [-52, 95, 5, -22, -21, 6, 28, -45, -7, -9, -22, -17, -93, 39, 1, -27, -52, 13, 58, 27, -25, -25, -30, 7, 33, 5, 16, -17, 68, 11, -12, 60, -50, -36, -89, 63, -73, -36, -30, -1, 35, 30, 19, 56, 41, -19, 11, -19, 4, -23, 34, -3, 2, -4, 2, -3, 2, -1, 1, -1, 1, 3, 1, -5, 0, -1, -4, -1, -3, 2, 0, 0]
assign weights = {8'sb11001100, 8'sb01011111, 8'sb00000101, 8'sb11101010, 8'sb11101011, 8'sb00000110, 8'sb00011100, 8'sb11010011, 8'sb11111001, 8'sb11110111, 8'sb11101010, 8'sb11101111, 8'sb10100011, 8'sb00100111, 8'sb00000001, 8'sb11100101, 8'sb11001100, 8'sb00001101, 8'sb00111010, 8'sb00011011, 8'sb11100111, 8'sb11100111, 8'sb11100010, 8'sb00000111, 8'sb00100001, 8'sb00000101, 8'sb00010000, 8'sb11101111, 8'sb01000100, 8'sb00001011, 8'sb11110100, 8'sb00111100, 8'sb11001110, 8'sb11011100, 8'sb10100111, 8'sb00111111, 8'sb10110111, 8'sb11011100, 8'sb11100010, 8'sb11111111, 8'sb00100011, 8'sb00011110, 8'sb00010011, 8'sb00111000, 8'sb00101001, 8'sb11101101, 8'sb00001011, 8'sb11101101, 8'sb00000100, 8'sb11101001, 8'sb00100010, 8'sb11111101, 8'sb00000010, 8'sb11111100, 8'sb00000010, 8'sb11111101, 8'sb00000010, 8'sb11111111, 8'sb00000001, 8'sb11111111, 8'sb00000001, 8'sb00000011, 8'sb00000001, 8'sb11111011, 8'sb00000000, 8'sb11111111, 8'sb11111100, 8'sb11111111, 8'sb11111101, 8'sb00000010, 8'sb00000000, 8'sb00000000};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [-2131, 902, -1547, 1077, 748, -73]
assign biases = {13'sb1011110101101, 13'sb0001110000110, 13'sb1100111110101, 12'sb010000110101, 12'sb001011101100, 12'sb111110110111};
top DUT(.inp(inp),
        .weights(weights),
        .biases(biases),
        .out(out)
        );


    integer inFile,outFile,i;
    initial
    begin
        $display($time, " << Starting the Simulation >>");
            inFile = $fopen("./sim/sim.Xtest","r");
        if (inFile == `NULL) begin
                $display($time, " file not found");
                $finish;
        end
        outFile = $fopen("./sim/output.txt");
        while (!$feof(inFile)) begin
            for (i=0;i<NUM_A;i=i+1) begin
                $fscanf(inFile,"%d ",at[i]);
            end
            $fscanf(inFile,"\n");
            #(period)
            $fwrite(outFile,"%d\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end


    genvar gi;
    generate
    for (gi=0;gi<NUM_A;gi=gi+1) begin : genbit
        assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
    end
    endgenerate


    endmodule
