import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.gui.WaitForUserDialog;
import ij.io.OpenDialog;
import java.io.File;
import java.io.PrintWriter;

public class disc_stabilization implements PlugIn {

	public void run(String arg) {

		OpenDialog od = new OpenDialog("Select first imagefile");
	
		String fileName = od.getFileName();

		File f = new File(od.getPath());
    		String path = f.getParent();
	
		String newPath = path+"\\pipeline1\\";

		File dir = new File(newPath);
		dir.mkdir();

		IJ.log("lastpathname:");
    		IJ.log(path.substring(path.lastIndexOf("\\")+1,path.length()));
		String finalPathName = newPath+path.substring(path.lastIndexOf("\\")+1,path.length())+".tif";
		IJ.log("willbesaved here:");
		IJ.log(finalPathName );

		IJ.run("Image Sequence...", "open="+od.getPath()+" sort");
		ImagePlus imp = IJ.getImage();

		
		//String path = od.getPath().substring(0,od.getPath().length()-imp.getTitle().length());
		//String imageName = imp.getTitle();
		//IJ.log(path);
		//IJ.log(imageName);

		IJ.setTool("Rectangle");

		new WaitForUserDialog("Selection required", "Please Select the optic disc with the rectangle tool flush to the edges.").show();

		Rectangle bounds = imp.getRoi().getBounds(); 
		IJ.log(Integer.toString(bounds.x));
		IJ.log(Integer.toString(bounds.y));
		IJ.log(Integer.toString(bounds.width));
		IJ.log(Integer.toString(bounds.height));

		//IJ.log(Integer.toString(bounds.x + bounds.width/2));
		//IJ.log(Integer.toString(bounds.y + bounds.height/2));
		//IJ.log(Integer.toString(bounds.x + bounds.width));
		//IJ.log(Integer.toString(bounds.y + bounds.height/2));

		String newMakeline = "makeLine("+Integer.toString(bounds.x + bounds.width/2)+","+Integer.toString(bounds.y + bounds.height/2)+","+Integer.toString(bounds.x + bounds.width)+","+Integer.toString(bounds.y + bounds.height/2)+");";
		IJ.log(newMakeline);

		try {
            		PrintWriter writer = new PrintWriter(newPath+"disc.txt", "UTF-8");
            		writer.println(newMakeline);
           		 	writer.close();
           		 }catch (Exception ex){ }

		IJ.run(imp, "Align slices in stack...", "method=5 windowsizex="+Integer.toString(bounds.width)+" windowsizey="+Integer.toString(bounds.height)+" x0="+Integer.toString(bounds.x)+" y0="+Integer.toString(bounds.y)+" swindow=0 subpixel=false itpmethod=0 ref.slice=1 show=true");
		IJ.run(imp, "Image Sequence... ", "format=TIFF use save="+finalPathName);
		IJ.run("Close All", "");

	}

}
