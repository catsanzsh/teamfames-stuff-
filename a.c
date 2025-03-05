#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

typedef struct {
    int input_dim;
    int hidden_dim;
    int style_dim;
    int output_dim;
    
    // RNN weights and biases with Xavier initialization
    float* W_h;    // shape: (hidden_dim, hidden_dim + input_dim + style_dim)
    float* b_h;    // shape: (hidden_dim)
    float* W_out;  // shape: (output_dim, hidden_dim)
    float* b_out;  // shape: (output_dim)
    
    // State variables
    float* h;           // hidden state: (hidden_dim)
    float* style_vector;// (style_dim)
    float* last_output; // (output_dim)
    
    // Pre-allocated temporary arrays
    float* combined_input;
    float* temp;
    
    // New: Training-related parameters
    float learning_rate;
    int initialized;
} TextureRNN;

// Helper functions with improved implementations
float randn(float mean, float stddev) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2); // Box-Muller transform
    return mean + stddev * z;
}

float activation(float x, const char* type) {
    if (strcmp(type, "sigmoid") == 0) return 1.0f / (1.0f + expf(-x));
    if (strcmp(type, "tanh") == 0) return tanhf(x);
    if (strcmp(type, "relu") == 0) return MAX(0.0f, x);
    return x; // linear by default
}

// Optimized matrix-vector multiplication
void matrix_vector_mult(const float* matrix, const float* vector, float* result,
                       int rows, int cols, float bias[], int use_bias) {
    #pragma omp parallel for if(rows > 100) // Parallelize for large matrices
    for (int i = 0; i < rows; i++) {
        float sum = use_bias ? bias[i] : 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

TextureRNN* texture_rnn_init(int input_dim, int hidden_dim, int style_dim, int output_dim, float lr) {
    TextureRNN* rnn = (TextureRNN*)calloc(1, sizeof(TextureRNN));
    if (!rnn) return NULL;
    
    // Set dimensions
    rnn->input_dim = input_dim;
    rnn->hidden_dim = hidden_dim;
    rnn->style_dim = style_dim;
    rnn->output_dim = output_dim;
    rnn->learning_rate = lr;
    rnn->initialized = 0;
    
    // Allocate memory with alignment
    int w_h_size = hidden_dim * (hidden_dim + input_dim + style_dim);
    rnn->W_h = (float*)aligned_alloc(16, w_h_size * sizeof(float));
    rnn->b_h = (float*)aligned_alloc(16, hidden_dim * sizeof(float));
    rnn->W_out = (float*)aligned_alloc(16, output_dim * hidden_dim * sizeof(float));
    rnn->b_out = (float*)aligned_alloc(16, output_dim * sizeof(float));
    rnn->h = (float*)calloc(hidden_dim, sizeof(float));
    rnn->style_vector = (float*)calloc(style_dim, sizeof(float));
    rnn->last_output = (float*)calloc(output_dim, sizeof(float));
    rnn->combined_input = (float*)calloc(hidden_dim + input_dim + style_dim, sizeof(float));
    rnn->temp = (float*)calloc(hidden_dim, sizeof(float));
    
    // Comprehensive allocation check
    if (!rnn->W_h || !rnn->b_h || !rnn->W_out || !rnn->b_out || 
        !rnn->h || !rnn->style_vector || !rnn->last_output || 
        !rnn->combined_input || !rnn->temp) {
        texture_rnn_free(rnn);
        return NULL;
    }
    
    // Xavier/Glorot initialization
    float fan_in = hidden_dim + input_dim + style_dim;
    float fan_out = hidden_dim;
    float std_h = sqrtf(6.0f / (fan_in + fan_out));
    for (int i = 0; i < w_h_size; i++) {
        rnn->W_h[i] = randn(0.0f, std_h);
    }
    
    fan_in = hidden_dim;
    fan_out = output_dim;
    float std_out = sqrtf(6.0f / (fan_in + fan_out));
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        rnn->W_out[i] = randn(0.0f, std_out);
    }
    
    rnn->initialized = 1;
    return rnn;
}

void texture_rnn_free(TextureRNN* rnn) {
    if (!rnn) return;
    free(rnn->W_h); free(rnn->b_h); free(rnn->W_out); free(rnn->b_out);
    free(rnn->h); free(rnn->style_vector); free(rnn->last_output);
    free(rnn->combined_input); free(rnn->temp);
    free(rnn);
}

void reset_state(TextureRNN* rnn, const float* style_vector, int reset_output) {
    if (!rnn || !rnn->initialized) return;
    
    memcpy(rnn->style_vector, style_vector, rnn->style_dim * sizeof(float));
    memset(rnn->h, 0, rnn->hidden_dim * sizeof(float));
    if (reset_output) {
        memset(rnn->last_output, 0, rnn->output_dim * sizeof(float));
    }
}

void rnn_step(TextureRNN* rnn, const float* inp, float* output_vec) {
    if (!rnn || !rnn->initialized) return;
    
    // Combine inputs
    memcpy(rnn->combined_input, rnn->h, rnn->hidden_dim * sizeof(float));
    memcpy(rnn->combined_input + rnn->hidden_dim, inp, rnn->input_dim * sizeof(float));
    memcpy(rnn->combined_input + rnn->hidden_dim + rnn->input_dim, 
           rnn->style_vector, rnn->style_dim * sizeof(float));
    
    // Hidden state update
    matrix_vector_mult(rnn->W_h, rnn->combined_input, rnn->temp,
                      rnn->hidden_dim, rnn->hidden_dim + rnn->input_dim + rnn->style_dim,
                      rnn->b_h, 1);
    for (int i = 0; i < rnn->hidden_dim; i++) {
        rnn->h[i] = activation(rnn->temp[i], "tanh");
    }
    
    // Output computation
    matrix_vector_mult(rnn->W_out, rnn->h, output_vec,
                      rnn->output_dim, rnn->hidden_dim, rnn->b_out, 1);
    for (int i = 0; i < rnn->output_dim; i++) {
        output_vec[i] = activation(output_vec[i], "sigmoid");
    }
}

// New: Simple training function
void train_step(TextureRNN* rnn, const float* input, const float* target) {
    if (!rnn || !rnn->initialized) return;
    
    float* output = (float*)malloc(rnn->output_dim * sizeof(float));
    rnn_step(rnn, input, output);
    
    // Simple gradient descent
    float error[rnn->output_dim];
    for (int i = 0; i < rnn->output_dim; i++) {
        error[i] = target[i] - output[i];
        rnn->b_out[i] += rnn->learning_rate * error[i];
        for (int j = 0; j < rnn->hidden_dim; j++) {
            rnn->W_out[i * rnn->hidden_dim + j] += rnn->learning_rate * error[i] * rnn->h[j];
        }
    }
    
    free(output);
}

float* generate_next_pixel(TextureRNN* rnn) {
    if (!rnn || !rnn->initialized) return NULL;
    
    float* output_vec = (float*)malloc(rnn->output_dim * sizeof(float));
    if (!output_vec) return NULL;
    
    rnn_step(rnn, rnn->last_output, output_vec);
    memcpy(rnn->last_output, output_vec, rnn->output_dim * sizeof(float));
    return output_vec;
}

void generate_texture_image(TextureRNN* rnn, int width, int height, const char* filename,
                          int channels) {
    if (!rnn || !rnn->initialized || width <= 0 || height <= 0 || !filename) return;
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return;
    }
    
    // Support different channel counts
    channels = MIN(channels, rnn->output_dim);
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    unsigned char* row_buffer = (unsigned char*)malloc(width * 3 * sizeof(unsigned char));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float* pixel = generate_next_pixel(rnn);
            if (!pixel) continue;
            
            for (int c = 0; c < 3; c++) {
                row_buffer[x * 3 + c] = (c < channels) ? 
                    (unsigned char)(pixel[c] * 255.0f) : 0;
            }
            free(pixel);
        }
        fwrite(row_buffer, sizeof(unsigned char), width * 3, fp);
    }
    
    free(row_buffer);
    fclose(fp);
    printf("Generated texture saved to %s (%dx%d)\n", filename, width, height);
}

int main() {
    srand(time(NULL));
    
    TextureRNN* rnn = texture_rnn_init(3, 128, 32, 3, 0.01f);
    if (!rnn) {
        fprintf(stderr, "Error: RNN initialization failed\n");
        return 1;
    }
    
    float style_vec[32] = {0};
    for (int i = 0; i < 32; i++) style_vec[i] = randn(0.0f, 0.1f);
    reset_state(rnn, style_vec, 1);
    
    generate_texture_image(rnn, 256, 256, "improved_texture.ppm", 3);
    
    texture_rnn_free(rnn);
    return 0;
}
