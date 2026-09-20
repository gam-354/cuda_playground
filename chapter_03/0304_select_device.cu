#include "../common/book.h"

int main( void ) {
    cudaDeviceProp prop;
    int dev;
    
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 3;

    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
    printf( "ID of CUDA device closest to revision %d.%d: %d\n", prop.major, prop.minor, dev );
    HANDLE_ERROR( cudaSetDevice( dev ) );
}