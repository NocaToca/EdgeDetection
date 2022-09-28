
public static class NMath{

    public static float pi = 3.141529f;
    public static float e = (float)Math.E;

    public static float Gaussian(float sigma, float x, float mu = 0.0f){

        float a = (float)(1.0f/Math.Sqrt(2.0f * pi)); 

        float b = -1.0f/2.0f * (float)Math.Pow((x - mu) / sigma, 2.0f);

        return a * (float)Math.Pow(e, b);

    }

}

public static class LinearAlgebra{

    public static Matrix Add(Matrix m1, Matrix m2){

        int width, height;
        width = (m1.width > m2.width) ? m2.width : m1.width;
        height = (m1.height > m2.height) ? m2.height : m1.height;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                m1.SetValue(y, x, m1.ValueAt(x,y) + m2.ValueAt(x,y));
            }
        }

        return m1;
    }

    public static Matrix Subtract(Matrix m1, Matrix m2){
        int width, height;
        width = (m1.width > m2.width) ? m2.width : m1.width;
        height = (m1.height > m2.height) ? m2.height : m1.height;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                m1.SetValue(y, x, m1.ValueAt(x,y) - m2.ValueAt(x,y));
            }
        }

        return m1;
    }

    public static Matrix Multiply(Matrix m1, Matrix m2){

        if(m1.width != m2.height){
            throw new ArgumentException("Matrix m1 and m2 cannot be multiplied together (m != n in nxm)");
        }

        Matrix result = new Matrix(m1.height, m2.width);
        int loops = m1.height;

        for(int y = 0; y < loops; y++){
            Vector v1 = m1.GetRowVector(y);
            for(int x = 0; x < m2.width; x++){
                Vector v2 = m2.GetColumnVector(x);

                result.SetValue(y, x, Dot(v1, v2));
            }
        }

        return result;
    }

    public static float Dot(Vector v1, Vector v2){

        float result = 0;

        for(int i = 0; i < v1.length; i++){
            result += v1.ValueAt(i) * v2.ValueAt(i);
        }

        return result;

    }

    public static Vector Cross(Vector v1, Vector v2){

        //Cross product is defined as:
        //u1 u2  u3
        //Where:
        //u1 = v1.y * v2.z - v1.z * v2.y
        //u2 = v1.z * v2.x - v1.x * v2.z
        //u3 = v1.x * v2.y - v1.y * v2.x

        //I'm assuming three dimensional vectors because there isnt much use to use the cross product outside of 3 dimensions (Unless you're working in 7 dimensions for some reason)
        if(v1.length != 3 || v2.length != 3){
            throw new ArgumentException();
        }

        float u1 = v1.ValueAt(1) * v2.ValueAt(2) - v1.ValueAt(2) * v2.ValueAt(1);
        float u2 = v1.ValueAt(2) * v2.ValueAt(0) - v1.ValueAt(0) * v2.ValueAt(2);
        float u3 = v1.ValueAt(0) * v2.ValueAt(1) - v1.ValueAt(1) * v2.ValueAt(0);

        Vector result = new Vector(3);
        float[] f = {u1,u2,u3};

        result.SetVector(f);
        return result; 
    }


}

public struct Vector{

    public int length;

    internal float[] vector;

    public Vector(int length){
        this.length = length;

        vector = new float[length];
        for(int i = 0; i < length; i++){
            vector[i] = 0;
        }
    }

    public void SetVector(float[] values){
        for(int i = 0; i < length; i++){
            vector[i] = values[i];
        }
    }

    public float ValueAt(int index){
        return vector[index];
    }

    public void SetValue(int index, float value){
        vector[index] = value;

    }

    public override string ToString(){
        
        string s = "";
        for(int i = 0; i < length; i ++){
            s += vector[i];
            if(i != length - 1){
                s+= ", ";
            }    
        }

        return s;
    }

    public static Vector CreateLinSpaced(int start, int end, int size){

        float stepsize = ((float)((end) - start))/(size-1);

        float location = start;
        Vector v = new Vector(size);

        for(int i = 0; i < size; i++){

            v.SetValue(i, location);
            location += stepsize;
        }

        return v;
    }


}

public struct Matrix{

    public int width;
    public int height;

    internal float[,] matrix;

    public Matrix(int height, int width){
        this.width = width;
        this.height = height;
        matrix = new float[height, width];
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                matrix[y,x] = 0.0f;
            }
        }
    }

    public Matrix Submatrix(int startrow, int endrow, int startcol, int endcol){

        int height = endrow - startrow;
        int width = endcol - startcol;

        Matrix output = new Matrix(height, width);

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                float value = ValueAt(y + startrow, x + startcol);
                output.SetValue(y,x,value);
            }
        }

        return output;

    }

    public float Sum(){
        float f = 0.0f;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                f += ValueAt(y,x);
            }
        }

        return f;
    }

    public Vector GetRowVector(int row){
        Vector v = new Vector(width);

        float[] vectorValues = new float[width];
        for(int i = 0; i < width; i++){
            vectorValues[i] = matrix[row, i];
        }

        v.SetVector(vectorValues);
        return v;
    }

    public Vector GetColumnVector(int column){
        Vector v = new Vector(height);

        float[] vectorValues = new float[height];
        for(int i = 0; i < height; i++){
            vectorValues[i] = matrix[i, column];
        }

        v.SetVector(vectorValues);
        return v;
    }

    public void SetValue(int row, int column, float value){
        matrix[row,column] = value;
    }

    public float ValueAt(int row, int column){
        return matrix[row, column];
    }

    public void SetRow(int row, float[] values){
        for(int x = 0; x < width; x++){
            matrix[row,x] = values[x];
        }
    }
    public void SetColumn(int column, float[] values){
        for(int y = 0; y < height; y++){
            matrix[y,column] = values[y];
        }
    }

    public float Max(){
        float max = float.NegativeInfinity;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                float f = matrix[y,x];
                if(f > max){
                    max = f;
                }
            }
        }

        return max;

    }

    public void Normalize(){
        float f = Max();
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                float value = ValueAt(y,x);
                SetValue(y,x, value/f);
            }
        }
    }

    public override string ToString(){

        string s = "";

        for(int i = 0; i < height; i ++){
            for(int j = 0; j < width; j++){
                s += matrix[i,j].ToString();
                if(j != width-1){
                    s += ", ";
                }
            }
            s += "\n";
        }

        return s;
    }

    public static Matrix OuterProduct(Vector v1, Vector v2){

        Matrix m = new Matrix(v1.length, v1.length);

        for(int y = 0; y < v1.length; y++){
            for(int x = 0; x < v1.length; x++){

                float value = v1.ValueAt(y) * v2.ValueAt(x);
                m.SetValue(y,x, value);
            }
        }

        return m;

    }

}