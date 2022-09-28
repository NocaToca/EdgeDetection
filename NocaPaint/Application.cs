// See https://aka.ms/new-console-template for more information
using System;
using System.Drawing;
using System.Windows.Forms;

public partial class WindowApplication : Form{

    private Button button;
    Task<Image> mainImageFilter;

    public WindowApplication() {
        DisplayGUI();
    }

    private void DisplayGUI() {
        this.Name = "WinForm Example";
        this.Text = "WinForm Example";
        this.Size = new Size(350, 350);
        this.StartPosition = FormStartPosition.CenterScreen;

        button = new Button();
        button.Name = "button";
        button.Text = "Click Me!";
        button.Size = new Size(150, 100);
        button.Location = new Point(
            (this.Width - button.Width) / 3 ,
            (this.Height - button.Height) / 3);
        button.Click += new System.EventHandler(this.MyButtonClick);

        this.Controls.Add(button);
    }

    private void MyButtonClick(object source, EventArgs e) {
        Image Luna = Image.FromFile("luna.jpg");
        Image filteredLuna = Edgedetection.Non_Max_Suppresion(Luna);

        filteredLuna.Save("luna_filtered.jpg"); 
    }

    public static void Main(){

        // Image Luna = Image.FromFile("ny.jpg");
        // Image filteredLuna = Edgedetection.Non_Max_Suppresion(Luna);

        // filteredLuna.Save("ny_filtered.jpg"); 

        Application.Run(new WindowApplication());

    }


}

