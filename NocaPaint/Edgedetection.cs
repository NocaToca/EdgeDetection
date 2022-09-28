using System;
using System.Drawing;
using System.Threading;
using System.Linq;

public class Edgedetection{

    public static Image CreateGrayscale(Image img){

        Bitmap originalImage = (Bitmap)img;

        Bitmap finalImage = new Bitmap(img.Width, img.Height);

        for(int y = 0; y < originalImage.Height; y++){
            for(int x = 0; x < originalImage.Width; x++){

                Color originalPixel = originalImage.GetPixel(x,y);

                int grayscaleValue = (int)(0.3f * originalPixel.R + 0.59 * originalPixel.G + 0.11 * originalPixel.B);
                Color grayscale = Color.FromArgb(grayscaleValue,grayscaleValue,grayscaleValue);

                finalImage.SetPixel(x,y, grayscale);

            }
        }

        Image result = (Image)finalImage;

        return result;

    }

    public static Matrix SobelX(){

        Matrix kernal = new Matrix(3,3);
        float[] r1 = {-1,-2,-1};
        float[] r2 = {0,0,0};
        float[] r3 = {1,2,1};
        kernal.SetRow(0,r1);
        kernal.SetRow(1,r2);
        kernal.SetRow(2,r3);

        return kernal;
    }

    public static Matrix SobelY(){

         Matrix kernal = new Matrix(3,3);
        float[] r1 = {-1,-2,-1};
        float[] r2 = {0,0,0};
        float[] r3 = {1,2,1};
        kernal.SetColumn(0,r1);
        kernal.SetColumn(1,r2);
        kernal.SetColumn(2,r3);

        return kernal;

    }


    /*
        So now we shall use our Math file so hopefully it works lmfao
    
    */
    /*public static Image CannyEdgedetection(Image img, float sigma = 1.0f, int threshold = 100){




    }*/

    public static Image Gaussian_Filter(Image image){

        Image grayscale = Edgedetection.CreateGrayscale(image);

        Bitmap bit = (Bitmap)grayscale;

        Matrix values = new Matrix(bit.Height, bit.Width);

        for(int y = 0; y < values.height; y++){
            for(int x = 0; x < values.width; x++){
                Color col = bit.GetPixel(x,y);
                values.SetValue(y,x,col.R);
            }
        }

        Matrix conv = Gaussian_Filter_Mat(values);

        conv.Normalize();

        Bitmap output = new Bitmap(conv.width, conv.height);
        // Console.WriteLine(output.Height);
        // Console.WriteLine(output.Width);
        // Console.WriteLine(conv.height);
        // Console.WriteLine(conv.width);
        for(int y = 0; y < output.Height; y++){
            for(int x = 0; x < output.Width; x++){
                float value = conv.ValueAt(y,x);

                int gray = 255 - (int)(value * 255.0f);

                Color col = Color.FromArgb(gray,gray,gray);
                
                output.SetPixel(x,y, col);
            }
        }

        return (Image)output;

    }

    public static Image Non_Max_Suppresion(Image image){

        Image grayscale = Edgedetection.CreateGrayscale(image);

        Bitmap bit = (Bitmap)grayscale;

        Matrix values = new Matrix(bit.Height, bit.Width);

        for(int y = 0; y < values.height; y++){
            for(int x = 0; x < values.width; x++){
                Color col = bit.GetPixel(x,y);
                values.SetValue(y,x,col.R);
            }
        }

        Matrix[] outputs = Sobel_Edge(values, 1, 100);

        outputs[0].Normalize();

        Matrix conv = Non_Max_Alg(outputs[0], outputs[1]);

        conv.Normalize();


        Bitmap output = new Bitmap(conv.width, conv.height);

        for(int y = 0; y < output.Height; y++){
            for(int x = 0; x < output.Width; x++){
                float value = conv.ValueAt(y,x);

                int gray = (int)(value * 255.0f);

                Color col = Color.FromArgb(gray,gray,gray);
                
                output.SetPixel(x,y, col);
            }
        }

        return (Image)output;

    }

    public static bool IsConnectedToStrongEdge(int y, int x, Matrix edges, int iterations){

        if(iterations > 500){
            return false;
        }

        if(y < edges.height - 1 && edges.ValueAt(y+1,x) == 2.0f){
            return true;
        }else
        if(x < edges.width - 1 && edges.ValueAt(y,x+1) == 2.0f){
            return true;
        }else
        if(y > 0 && edges.ValueAt(y-1,x) == 2.0f){
            return true;
        }else
        if(x > 0 && edges.ValueAt(y, x - 1) == 2.0f){
            return true;
        }else
        if(y < edges.height-1 && x < edges.width-1 && edges.ValueAt(y+1,x+1) == 2.0f){
            return true;
        }else
        if(y < edges.height-1 && x > 0 && edges.ValueAt(y+1,x-1) == 2.0f){
            return true;
        }else
        if(y > 0 && x < edges.width-1 && edges.ValueAt(y-1, x+1) == 2.0f){
            return true;
        }else
        if(y > 0 && x > 0 && edges.ValueAt(y-1,x-1) == 2.0f){
            return true;
        }

        //Now we can see if its connected to another weak edge
        bool connected = false;

        if(y < edges.height - 1 && edges.ValueAt(y+1,x) != 0){
            connected = connected || IsConnectedToStrongEdge(y+1, x, edges, iterations+1);
        }
        if(!connected && x < edges.width - 1 && edges.ValueAt(y,x+1) != 0){
            connected = connected || IsConnectedToStrongEdge(y, x+1, edges, iterations+1);
        }
        if(!connected && y > 0 && edges.ValueAt(y-1,x) != 0){
            connected = connected || IsConnectedToStrongEdge(y-1, x, edges, iterations+1);
        }
        if(!connected && x > 0 && edges.ValueAt(y, x - 1) != 0){
            connected = connected || IsConnectedToStrongEdge(y, x-1, edges, iterations+1);
        }
        if(!connected && y < edges.height-1 && x < edges.width-1 && edges.ValueAt(y+1,x+1) != 0){
            connected = connected || IsConnectedToStrongEdge(y+1, x+1, edges, iterations+1);
        }
        if(!connected && y < edges.height-1 && x > 0 && edges.ValueAt(y+1,x-1) != 0){
            connected = connected || IsConnectedToStrongEdge(y+1, x-1, edges, iterations+1);
        }
        if(!connected && y > 0 && x < edges.width-1 && edges.ValueAt(y-1, x+1) != 0){
            connected = connected || IsConnectedToStrongEdge(y-1, x+1, edges, iterations+1);
        }
        if(!connected && y > 0 && x > 0 && edges.ValueAt(y-1,x-1) != 0){
            connected = connected || IsConnectedToStrongEdge(y-1, x-1, edges, iterations+1);
        }

        return connected;

    }

    public static Matrix Double_Thresholding(Matrix image, float highThresholdRatio = 0.7f, float lowThresholdRatio = 0.3f){

        float highThreshold = image.Max() * highThresholdRatio;
        float lowThreshold = highThreshold * lowThresholdRatio;

        Matrix output = new Matrix(image.height, image.width);

        Matrix edges = new Matrix(image.height, image.width);

        for(int y = 0; y < image.height; y++){
            for(int x = 0; x < image.width; x++){

                if(image.ValueAt(y,x) > lowThreshold && image.ValueAt(y,x) < highThreshold){
                    edges.SetValue(y,x,2.0f);
                } else
                if(image.ValueAt(y,x) >= highThreshold){
                    edges.SetValue(y,x,1.0f);
                }

            }
        }

        Console.WriteLine("Finished Determining Weak and Strong Edges");
        
        for(int y = 0; y < image.height; y++){
            for(int x = 0; x < image.width; x++){

                if(IsConnectedToStrongEdge(y,x, edges, 0)){
                    output.SetValue(y,x,image.ValueAt(y,x));
                }

            }
            Console.WriteLine("Finished row " + y + " of recursion");
        }

        return output;

    }

    public static Matrix Non_Max_Alg(Matrix image, Matrix angles){

        int image_x = image.width;
        int image_y = image.height;

        Matrix output = new Matrix(image_y,image_x);

        for(int y = 1; y < image_y-1; y++){
            for(int x = 1; x < image_x-1; x++){

                if((angles.ValueAt(y,x) >= -22.5f && angles.ValueAt(y,x) <= 22.5f)
                    || (angles.ValueAt(y,x) < -157.5f && angles.ValueAt(y,x) >= 180.0f)){
                    if((image.ValueAt(y,x) >= image.ValueAt(y,x+1)) && (image.ValueAt(y,x) >= image.ValueAt(y,x-1))){

                        output.SetValue(y,x,image.ValueAt(y,x));

                    }
                } else
                if((angles.ValueAt(y,x) >= 22.5f && angles.ValueAt(y,x) <= 67.5f)
                    || (angles.ValueAt(y,x) < -122.5 && angles.ValueAt(y,x) >= -157.5f)){
                    
                    if((image.ValueAt(y,x) >= image.ValueAt(y+1,x+1)) && (image.ValueAt(y,x) >= image.ValueAt(y-1,x-1))){

                        output.SetValue(y,x,image.ValueAt(y,x));

                    }

                }else
                if((angles.ValueAt(y,x) >= 67.5f && angles.ValueAt(y,x) <= 112.5f)
                    || (angles.ValueAt(y,x) < -67.5 && angles.ValueAt(y,x) >= -112.5f)){
                    
                    if((image.ValueAt(y,x) >= image.ValueAt(y+1,x)) && (image.ValueAt(y,x) >= image.ValueAt(y-1,x))){

                        output.SetValue(y,x,image.ValueAt(y,x));

                    }

                }else
                if((angles.ValueAt(y,x) >= 112.5f && angles.ValueAt(y,x) <= 157.5f)
                    || (angles.ValueAt(y,x) < -22.5 && angles.ValueAt(y,x) >= -67.5f)){
                    
                    if((image.ValueAt(y,x) >= image.ValueAt(y+1,x-1)) && (image.ValueAt(y,x) >= image.ValueAt(y-1,x+1))){

                        output.SetValue(y,x,image.ValueAt(y,x));

                    }

                }

            }
        }

        Console.WriteLine("Finished Non-Max-Supression");

        return output;

    }

    public static Image Sobel_Filter(Image image){

        Image grayscale = Edgedetection.CreateGrayscale(image);

        Bitmap bit = (Bitmap)grayscale;

        Matrix values = new Matrix(bit.Height, bit.Width);

        for(int y = 0; y < values.height; y++){
            for(int x = 0; x < values.width; x++){
                Color col = bit.GetPixel(x,y);
                values.SetValue(y,x,col.R);
            }
        }

        Matrix conv = Sobel_Edge(values)[0];
        conv.Normalize();

        Bitmap output = new Bitmap(conv.width, conv.height);
        for(int y = 0; y < output.Height; y++){
            for(int x = 0; x < output.Width; x++){
                float value = conv.ValueAt(y,x);

                Color col = Color.FromArgb((int)(value * 255), (int)(value * 255), (int)(value * 255));
                
                output.SetPixel(x,y, col);
            }
        }

        return (Image)output;

    }

    public static Matrix Gaussian_Filter_Mat(Matrix image){



        Matrix kernal = Edgedetection.Gaussian_Kernal(1.0f);

        Matrix conv = Edgedetection.Convolution(image, kernal);

        conv.Normalize();

        return conv;

    }

    public static Matrix[] Sobel_Edge(Matrix image, float sigma = 1.0f, int threshold = 100){

        Matrix smoothed = Gaussian_Filter_Mat(image);

        Matrix kernalX = Edgedetection.SobelX();
        Matrix kernalY = Edgedetection.SobelY();

        

        Task<Matrix> threadX = Task<Matrix>.Factory.StartNew(() => {
            return Convolution(smoothed, kernalX);
        });

        Task<Matrix> threadY = Task<Matrix>.Factory.StartNew(() => {
            return Convolution(smoothed, kernalX);
        });

        threadX.Wait();
        threadY.Wait();

        Matrix sobelX = threadX.Result;
        Matrix sobelY = threadY.Result;

        Matrix gradient_magnitude = new Matrix(sobelX.height, sobelX.width);

        for(int y = 0; y < sobelX.height; y++){
            for(int x = 0; x < sobelX.width; x++){

                float a = sobelX.ValueAt(y,x);
                float b = sobelY.ValueAt(y,x);

                a *= a;
                b *= b;
                float values = (float)Math.Sqrt(a + b);
                gradient_magnitude.SetValue(y,x,values);

            }
        }

        float value = threshold/gradient_magnitude.Max();

        for(int y = 0; y < sobelX.height; y++){
            for(int x = 0; x < sobelX.width; x++){

                gradient_magnitude.SetValue(y,x, gradient_magnitude.ValueAt(y,x) * value);

            }
        }

        Matrix[] outputs = new Matrix[2];
        outputs[0] = gradient_magnitude;

        Matrix angles = new Matrix(gradient_magnitude.height, gradient_magnitude.width);

        for(int y = 0; y < angles.height; y++){
            for(int x = 0; x < angles.width; x++){
                float a = sobelX.ValueAt(y,x);
                float b = sobelY.ValueAt(y,x);

                float val = (float)Math.Atan2(b,a) * 180.0f/NMath.pi;

                angles.SetValue(y,x,val);
            }
        }

        Console.WriteLine("Finished Angles Matrix");

        outputs[1] = angles;

        return outputs;
    }

    public static Matrix Gaussian_Kernal(float sigma, int size = 5){

        int a = size / 2;

        Vector vectorKernal = Vector.CreateLinSpaced(-a, a, size);

        for(int i = 0; i < size; i++){
            vectorKernal.SetValue(i, NMath.Gaussian(sigma, vectorKernal.ValueAt(i), 0));
        }

        Matrix kernal = Matrix.OuterProduct(vectorKernal, vectorKernal);

        float maxValue = kernal.Max();

        float mul = 1.0f/maxValue;

        Matrix newKernal = new Matrix(size, size);
        for(int y = 0; y < size; y++){
            for(int x = 0; x < size; x++){
                float value = kernal.ValueAt(y,x);
                value = value * mul;
                newKernal.SetValue(y,x,value);
            }
        }

        Console.WriteLine("Finished Making Kernal");

        return newKernal;
    }

    public static Matrix Convolution(Matrix image, Matrix kernal){

        int image_x = image.width;
        int image_y = image.height;

        int kernal_x = kernal.width;
        int kernal_y = kernal.height;

        Matrix output = new Matrix(image_y, image_x);

        int padding_x = (kernal_x - 1)/2;
        int padding_y = (kernal_y - 1)/2;

        int pad_image_x = image_x + (2 * padding_x);
        int pad_image_y = image_y + (2 * padding_y);


        Matrix paddedImage = new Matrix(pad_image_y, pad_image_x);
        
        for(int y = 0; y < image_y; y++){
            for(int x = 0; x < image_x; x++){

                float value = image.ValueAt(y,x);
                paddedImage.SetValue(y+padding_y, x+padding_x, value);
            }
        }

        for(int y = 0; y < pad_image_y; y++){
            for(int x = 0; x <pad_image_x; x++){

                if(x < padding_x && y > padding_y && y < image_y){
                    float value = image.ValueAt(padding_y, x);
                    paddedImage.SetValue(y,x,value);
                }else

                if(x < padding_x && y < padding_y){
                    float value = image.ValueAt(0,0);
                    paddedImage.SetValue(y,x,value);
                }else

                if(x > padding_x && y < padding_y && x < image_x){
                    float value = image.ValueAt(padding_y,x);
                    paddedImage.SetValue(y,x,value);
                }else

                if(x > padding_x && y >= image_y && x < image_x){
                    float value = image.ValueAt(image_y-1,x);
                    paddedImage.SetValue(y,x,value);
                }else

                if(x >= image_x && y < padding_y){
                    float value = image.ValueAt(0, image_x-1);
                    paddedImage.SetValue(y,x,value);
                }else

                if(x >= image_x && y >= image_y){
                    float value = image.ValueAt(image_y-1, image_x-1);
                    paddedImage.SetValue(y,x,value);
                }else

                if(x < padding_y && y >= image_y){
                    float value = image.ValueAt(image_y - 1, 0);
                    paddedImage.SetValue(y,x,value);
                }else

                if(x >= image_x && y > padding_y && y < image_y){
                    float value = image.ValueAt(y, image_x - 1);
                    paddedImage.SetValue(y,x,value);
                }

            }
        }

        Console.WriteLine("Finished Making Padded Image");

        for(int y = 0; y < image_y; y++){
            for(int x = 0; x< image_x; x++){
                int px = x + kernal_x;
                int py = y + kernal_y;

                Matrix sub = paddedImage.Submatrix(y, py, x, px);
                float value = LinearAlgebra.Multiply(sub, kernal).Sum();

                output.SetValue(y,x,value);
            }
        }

        Console.WriteLine("Finished Convoluting");

        return output;

    }

}