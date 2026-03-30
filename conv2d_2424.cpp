#include "mix_40300.h"



//新輸入
#define c1_c 24
//輸入權重
#define c2_c 24 //輸出通道

void compute_outputtt(
	fixed_t local_input_part11[bb][c1/2][input_height][input_width],
	fixed_t local_input_part22[bb][c1/2][input_height][input_width],
	fixed_ttt local_weight_part1[c2_c][c1/2][kernel_size_a][kernel_size_b],
	fixed_ttt local_weight_part2[c2_c][c1/2][kernel_size_a][kernel_size_b],
	float local_output[bb][c2][output_height][output_width],
    int start_channel,
    int end_channel

) {

	 for (int c_out = 0; c_out < c2_c; ++c_out) {
		 	#pragma HLS UNROLL //將迴圈計算展開
	        for (int i = 0; i < output_height; ++i) {
	            for (int j = 0; j < output_width; ++j) {
	            	fixed_tt sum = 0;
	                for (int c_in = start_channel; c_in < end_channel; ++c_in) {
	                            if (c_in < c1_c/2) {
	                                sum += local_input_part11[0][c_in][i][j] * local_weight_part1[c_out ][c_in][0][0];
	                            } else {
	                                sum += local_input_part22[0][c_in - c1_c/2][i][j] * local_weight_part2[c_out ][c_in - c1_c/2][0][0];
	                    }
	                }
	                local_output[0][c_out][i][j] = sum;
	            }
	        }
	    }
}


void conv2d_2424(
		fixed_t local_input_part11[bb][c1/2][input_height][input_width],
		fixed_t local_input_part22[bb][c1/2][input_height][input_width],
		fixed_ttt local_weight_part1[c2][c1/2][kernel_size_a][kernel_size_b],
		fixed_ttt local_weight_part2[c2][c1/2][kernel_size_a][kernel_size_b],
		float local_output1[bb][c2][output_height][output_width],
		float local_output2[bb][c2][output_height][output_width],
		float local_output3[bb][c2][output_height][output_width],
		float local_output4[bb][c2][output_height][output_width]

){
    #pragma HLS DATAFLOW //將下列四個計算進行同時處理 
    compute_outputtt(local_input_part11, local_input_part22, local_weight_part1, local_weight_part2, local_output1, 0, c1_c/4);
    compute_outputtt(local_input_part11, local_input_part22, local_weight_part1, local_weight_part2, local_output2, c1_c/4, c1_c/2);
    compute_outputtt(local_input_part11, local_input_part22, local_weight_part1, local_weight_part2, local_output3, c1_c/2, 3*c1_c/4);
    compute_outputtt(local_input_part11, local_input_part22, local_weight_part1, local_weight_part2, local_output4, 3*c1_c/4, c1_c);
}





