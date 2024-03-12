`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=4;
    parameter NUM_A=16;
    parameter WIDTH_A=4;
    parameter NUM_W=78;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=9;
    parameter NUM_B1=10;
    parameter WIDTH_B1=15;

localparam period = 320000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [9, -5, -2, -30, 1, 0, -4, 7, -17, -11, -17, 12, 11, 9, -2, -7, -73, 9, 16, 10, -45, -32, -17, -36, 0, -32, -32, -19, -4, 7, -53, -11, -31, 7, 24, 7, -14, 23, 20, -16, 26, -16, -5, 19, 0, 12, 79, 0, 16, -4, -17, 9, 13, -7, -26, -6, -6, 27, 17, -5, 12, -14, 31, -13, 37, 6, 0, 36, -23, -12, 10, 0, 2, -44, -41, 18, 32, 16]
assign weights = {8'sb00001001, 8'sb11111011, 8'sb11111110, 8'sb11100010, 8'sb00000001, 8'sb00000000, 8'sb11111100, 8'sb00000111, 8'sb11101111, 8'sb11110101, 8'sb11101111, 8'sb00001100, 8'sb00001011, 8'sb00001001, 8'sb11111110, 8'sb11111001, 8'sb10110111, 8'sb00001001, 8'sb00010000, 8'sb00001010, 8'sb11010011, 8'sb11100000, 8'sb11101111, 8'sb11011100, 8'sb00000000, 8'sb11100000, 8'sb11100000, 8'sb11101101, 8'sb11111100, 8'sb00000111, 8'sb11001011, 8'sb11110101, 8'sb11100001, 8'sb00000111, 8'sb00011000, 8'sb00000111, 8'sb11110010, 8'sb00010111, 8'sb00010100, 8'sb11110000, 8'sb00011010, 8'sb11110000, 8'sb11111011, 8'sb00010011, 8'sb00000000, 8'sb00001100, 8'sb01001111, 8'sb00000000, 8'sb00010000, 8'sb11111100, 8'sb11101111, 8'sb00001001, 8'sb00001101, 8'sb11111001, 8'sb11100110, 8'sb11111010, 8'sb11111010, 8'sb00011011, 8'sb00010001, 8'sb11111011, 8'sb00001100, 8'sb11110010, 8'sb00011111, 8'sb11110011, 8'sb00100101, 8'sb00000110, 8'sb00000000, 8'sb00100100, 8'sb11101001, 8'sb11110100, 8'sb00001010, 8'sb00000000, 8'sb00000010, 8'sb11010100, 8'sb11010111, 8'sb00010010, 8'sb00100000, 8'sb00010000};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [-7548, -2146, -3588, -1215, -11078, 551, -10964, 8870, 1446, -2150, -29, -192, 22]
assign biases = {15'sb110001010000100, 15'sb111011110011110, 15'sb111000111111100, 15'sb111101101000001, 15'sb101010010111010, 15'sb000001000100111, 15'sb101010100101100, 15'sb010001010100110, 15'sb000010110100110, 15'sb111011110011010, 9'sb111100011, 9'sb101000000, 9'sb000010110};
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
