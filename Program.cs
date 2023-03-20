
using System;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
//using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
namespace Benford
{
    public class BenfordsLaw
    {
    	public double[] arr;
        public double[,] multiarr;
    	public int[] distributions;
        MemoryBuffer1D<int, Stride1D.Dense> distBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> arrBuffer;
        

        MemoryBuffer1D<double, Stride1D.Dense> valsbuffer;
        



        Context context;
        Device dev;


        Accelerator accelerate;
    	public BenfordsLaw(double[] array)
        {
           	this.arr = array;
            this.distributions = new int[10];
            this.context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());

            this.dev = this.context.GetPreferredDevice(preferCPU: false);
            

        }
        public void setMultiArr(double[,] arr){
            this.multiarr = arr;
        }
        public double[] mutliBenfordNoGPU(){
            double[] output = new double[this.multiarr.GetLength(1)];
            double x;
            double y;

            double xsqr;
            double ysqr;
            double ben;
            double arrval;

            double xy;
            int num;
            int n;
 
            for(int k =0; k< multiarr.GetLength(1); k++){

                for(int i = 0; i < this.distributions.GetLength(0); i++){
                    this.distributions[i] = 0;
                }
                for(int i = 0; i < this.multiarr.GetLength(0); i++){
                    num =  (int)this.multiarr[i,k];
                    n = (int)this.multiarr[i,k];
                    while (num >= 10)
                    {
                        num = num / 10;

                    }

                    this.distributions[num] += 1;
        
                }
                x = 0;
                y = 0;
                xy = 0;
                xsqr = 0;
                ysqr = 0;
                ben = 0.0;
                arrval = 0.0;
                for(int i = 1; i < this.distributions.GetLength(0); i++){
                    ben = calcBenford(i);
      
                    arrval = (double)this.distributions[i]/this.multiarr.GetLength(0);
            
                    x +=arrval;
                    y +=ben;
                    xy += (arrval *ben);
                    xsqr += (arrval * arrval);
                    ysqr += (ben * ben);
                }

                output[k] = ( (9*xy) - (x*y))/Math.Sqrt(((9*xsqr) - (x*x))*((9*ysqr) - (y*y)));
            }
            return output;

        }
        public int[] getDist(){
        	int num;
            int n;

        	for(int i = 0; i < this.arr.GetLength(0); i++){
        		num =  (int)this.arr[i];
                n = (int)this.arr[i];
        		while (num >= 10)
		        {
		            num = num / 10;

		        }
             
                
		        this.distributions[num] += 1;
		  
        	}
        	return this.distributions;

        }
        public double calcBenford(double x){
        	return Math.Log10(1+(1/x));
        }
        //R = n(∑xy) – (∑x)(∑y) / √[n∑x²-(∑x)²][n∑y²-(∑y)²
        private double linearCorrelation(){

        	double x = 0;
        	double y = 0;
        	double xy = 0;
        	double xsqr = 0;
        	double ysqr = 0;
        	double ben = 0.0;
        	double arrval = 0.0;
        	for(int i = 1; i < this.distributions.GetLength(0); i++){
        		ben = calcBenford(i);
        
        		arrval = (double)this.distributions[i]/this.arr.GetLength(0);
        
        		x +=arrval;
        		y +=ben;
        		xy += (arrval *ben);
        		xsqr += (arrval * arrval);
        		ysqr += (ben * ben);
        	}

            
        	return ((9*xy) - (x*y))/Math.Sqrt(((9*xsqr) - (x*x))*((9*ysqr) - (y*y)));

        }
        private double linearCorrelationGPU(){
            accelerate = this.dev.CreateAccelerator(this.context);
            this.distBuffer = accelerate.Allocate1D<int>(new Index1D(10));
            this.valsbuffer = accelerate.Allocate1D<double>(new Index1D(5));

            this.arrBuffer =accelerate.Allocate1D<double>(new Index1D(this.arr.GetLength(0)));
            

            this.arrBuffer.CopyFromCPU(this.arr);

            var getDistKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                        
                ArrayView1D<int, Stride1D.Dense>>(getDistKernal); 
            var linearCorrKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                         
                ArrayView1D<double, Stride1D.Dense>,int>(linearCorrKernal); 
            var setBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueKernal);
            var setIntBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int>(
                setIntBuffToValueKernal);
            setIntBuffToValueKern(this.distBuffer.Extent.ToIntIndex(), this.distBuffer.View, 0);
            setBuffToValueKern(this.valsbuffer.Extent.ToIntIndex(), this.valsbuffer.View, 0.0);
            

            getDistKern(this.arrBuffer.Extent.ToIntIndex(), this.arrBuffer.View,this.distBuffer.View);
            
            linearCorrKern(this.distBuffer.Extent.ToIntIndex(), this.distBuffer.View, this.valsbuffer.View,this.arr.GetLength(0));
           
            double[] valarr = this.valsbuffer.GetAsArray1D();
            double x= valarr[0];
            double y= valarr[1];

            double xsqr= valarr[2];
            double ysqr= valarr[3];

            double xy= valarr[4];

           
            return ((9*xy) - (x*y))/Math.Sqrt(((9*xsqr) - (x*x))*((9*ysqr) - (y*y)));


        }
        private double hybrid(){
            accelerate = this.dev.CreateAccelerator(this.context);
            this.distBuffer = accelerate.Allocate1D<int>(new Index1D(10));
            this.valsbuffer = accelerate.Allocate1D<double>(new Index1D(5));

            this.arrBuffer =accelerate.Allocate1D<double>(new Index1D(this.arr.GetLength(0)));
            

            this.arrBuffer.CopyFromCPU(this.arr);

            var getDistKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                        
                ArrayView1D<int, Stride1D.Dense>>(getDistKernal); 
            var linearCorrKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                         
                ArrayView1D<double, Stride1D.Dense>,int>(linearCorrKernal); 
            var setBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueKernal);
            var setIntBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int>(
                setIntBuffToValueKernal);
            setIntBuffToValueKern(this.distBuffer.Extent.ToIntIndex(), this.distBuffer.View, 0);
            setBuffToValueKern(this.valsbuffer.Extent.ToIntIndex(), this.valsbuffer.View, 0.0);
            

            getDistKern(this.arrBuffer.Extent.ToIntIndex(), this.arrBuffer.View,this.distBuffer.View);
            this.distributions = this.distBuffer.GetAsArray1D();
            
            return linearCorrelation();
        }
        private double[] mutliBenford(){
            double[] output = new double[this.multiarr.GetLength(1)];
            double x;
            double y;

            double xsqr;
            double ysqr;

            double xy;

           
            accelerate = this.dev.CreateAccelerator(this.context);
            var multiGetDistKern = accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                        
                ArrayView2D<int, Stride2D.DenseX>>(multiGetDistKernal); 
            var multiLinearCorrKern = accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<int, Stride2D.DenseX>,
                         
                ArrayView2D<double, Stride2D.DenseX>,int>(multiLinearCorrKernal); 
            var setBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                double>(
                setBuffToValue2DKernal);
            var setIntBuffToValue2DKern = accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<int, Stride2D.DenseX>,
                int>(
                setIntBuffToValue2DKernal);

            var finalEqKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView2D<double, Stride2D.DenseX>,         
                ArrayView1D<double, Stride1D.Dense>>(finalEqKernal); 

            var MultiBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(this.multiarr.GetLength(0), this.multiarr.GetLength(1)));
            var MultiDistBuffer = accelerate.Allocate2DDenseX<int>(new Index2D(this.multiarr.GetLength(1), 10));
            var MutliValsBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(this.multiarr.GetLength(1), 5));
            var OutputBuffer = accelerate.Allocate1D<double>(new Index1D(this.multiarr.GetLength(1)));


            MultiBuffer.CopyFromCPU(this.multiarr);

        
            
            setIntBuffToValue2DKern(MultiDistBuffer.Extent.ToIntIndex(), MultiDistBuffer.View, 0);
            setBuffToValue2DKern(MutliValsBuffer.Extent.ToIntIndex(), MutliValsBuffer.View, 0.0);
            

            multiGetDistKern(MultiBuffer.Extent.ToIntIndex(), MultiBuffer.View,MultiDistBuffer.View);
            
            multiLinearCorrKern(MultiDistBuffer.Extent.ToIntIndex(), MultiDistBuffer.View, MutliValsBuffer.View,this.arr.GetLength(0));
            finalEqKern(OutputBuffer.Extent.ToIntIndex(), MutliValsBuffer.View, OutputBuffer.View);
          
            
            return OutputBuffer.GetAsArray1D();
            
        }
        void print1d(int[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");

        }
        void print1d(double[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");

        }
        //(int)(n / (int)(Math.pow(10, digits)))
        static void getDistKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> totalView,
            

            ArrayView1D<int, Stride1D.Dense> distView
            )
        {
            ///<summary>Fills r and rr using bView</summary>
            ///<param name="bView">bView</param>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>
            
            int num = (int)totalView[index];
            while (num >= 10)
            {
                num = num / 10;

            }
            
            Atomic.Add(ref distView[num], 1);

            
        }
        static void finalEqKernal(Index1D index,
            ArrayView2D<double, Stride2D.DenseX> MainView,
            ArrayView1D<double, Stride1D.Dense> outputView
            ){
            int i = index.X;
            double x= MainView[new Index2D(i,0)];
            double y= MainView[new Index2D(i,1)];

            double xsqr= MainView[new Index2D(i,2)];
            double ysqr= MainView[new Index2D(i,3)];

            double xy= MainView[new Index2D(i,4)];
            outputView[index] = ((9*xy) - (x*y))/Math.Sqrt(((9*xsqr) - (x*x))*((9*ysqr) - (y*y)));
        }
        static void multiGetDistKernal(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> totalView,
            

            ArrayView2D<int, Stride2D.DenseX> distView
            )
        {
            ///<summary>Fills r and rr using bView</summary>
            ///<param name="bView">bView</param>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>
            
            int num = (int)totalView[index];
            while (num >= 10)
            {
                num = num / 10;

            }
            
            Atomic.Add(ref distView[new Index2D(index.Y, num)], 1);

            
        }
        static void linearCorrKernal(Index1D index,
            ArrayView1D<int, Stride1D.Dense> distView,
            ArrayView1D<double, Stride1D.Dense> valView,

            int arrlength
            )
        {
            ///<summary>Fills r and rr using bView</summary>
            ///<param name="bView">bView</param>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>
            if(index.X > 0){
                double arrval = (double)distView[index]/arrlength;
                double ben = Math.Log10(1+(1/(double)index.X));
                Atomic.Add(ref valView[0], arrval);
                Atomic.Add(ref valView[1], ben);

                Atomic.Add(ref valView[4], (arrval *ben));
                Atomic.Add(ref valView[2], (arrval * arrval));

                Atomic.Add(ref valView[3], (ben * ben));
                

            }
            
        }
        static void multiLinearCorrKernal(Index2D index,
            ArrayView2D<int, Stride2D.DenseX> distView,
            ArrayView2D<double, Stride2D.DenseX> valView,

            int arrlength
            )
        {
            ///<summary>Fills r and rr using bView</summary>
            ///<param name="bView">bView</param>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>
            if(index.Y > 0){
                double arrval = (double)distView[index]/arrlength;
                double ben = Math.Log10(1+(1/(double)index.Y));
                Atomic.Add(ref valView[new Index2D(index.X,0)], arrval);
                Atomic.Add(ref valView[new Index2D(index.X,1)], ben);

                Atomic.Add(ref valView[new Index2D(index.X,4)], (arrval *ben));
                Atomic.Add(ref valView[new Index2D(index.X,2)], (arrval * arrval));

                Atomic.Add(ref valView[new Index2D(index.X,3)], (ben * ben));
                

            }
            
        }
        static void setBuffToValueKernal(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> buff, 
            double setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void setBuffToValue2DKernal(Index2D index, 
            ArrayView2D<double, Stride2D.DenseX> buff, 
            double setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void setIntBuffToValueKernal(Index1D index, 
            ArrayView1D<int, Stride1D.Dense> buff, 
            int setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void setIntBuffToValue2DKernal(Index2D index, 
            ArrayView2D<int, Stride2D.DenseX> buff, 
            int setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void Main(string[] args)
        {
        	Random rand = new Random();
        	double[] test= new double[1000000];//{423.1, 2435443.3, 64.3, 773.3, 4.35};
            double[,] test2 = new double[1000000,50];
        	for(int i = 0; i < test.GetLength(0); i++){
        		test[i] = rand.Next(1, 100000);
        	}
            for(int i = 0; i < test2.GetLength(0); i++){
                for(int j = 0; j < test2.GetLength(1); j++){
                    test2[i,j] = rand.Next(1, 100000);
                }
            }
        	BenfordsLaw x = new BenfordsLaw(test);
            x.setMultiArr(test2);
            //x.print1d(test);
            Console.WriteLine("BENFORD");
            Stopwatch stop = new Stopwatch();
            stop.Start();
            x.getDist();

            Console.WriteLine(x.linearCorrelation());
            for(int i = 0; i < 15; i++){
                x.getDist();
                x.linearCorrelation();

            }
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);
            Console.WriteLine("TEST");
            stop.Reset();
            stop.Start();

            //Console.WriteLine(x.linearCorrelationGPU());
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);

            stop.Reset();
            stop.Start();

            //Console.WriteLine(x.hybrid());
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);

            Console.WriteLine("MULTI");
            stop.Reset();
            stop.Start();
            x.print1d(x.mutliBenford());
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);
            Console.WriteLine();
            stop.Reset();
            stop.Start();
            x.print1d(x.mutliBenfordNoGPU());
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);
            Console.WriteLine();


            //x.linearCorrelationGPU();
        	//int[] output = x.getDist();
        	// for(int i = 1; i < 10; i++){
        	// 	Console.WriteLine(x.calcBenford(i));
        	// }
        }
    }
}
