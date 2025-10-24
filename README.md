private void InitializeComponent()
{
    System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(CustomMsgBox));
    this.button1 = new System.Windows.Forms.Button();
    this.progressBar1 = new System.Windows.Forms.ProgressBar();
    this.label1 = new System.Windows.Forms.Label();
    this.SuspendLayout();
    // 
    // button1
    // 
    this.button1.Font = new System.Drawing.Font("Calibri", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
    this.button1.Location = new System.Drawing.Point(233, 96);
    this.button1.Name = "button1";
    this.button1.Size = new System.Drawing.Size(112, 29);
    this.button1.TabIndex = 0;
    this.button1.Text = "Stop Trace";
    this.button1.UseVisualStyleBackColor = true;
    this.button1.Click += new System.EventHandler(this.button1_Click);
    // 
    // progressBar1
    // 
    this.progressBar1.ForeColor = System.Drawing.Color.Green;
    this.progressBar1.Location = new System.Drawing.Point(15, 34);
    this.progressBar1.MarqueeAnimationSpeed = 10;
    this.progressBar1.Maximum = 1000;
    this.progressBar1.Name = "progressBar1";
    this.progressBar1.Size = new System.Drawing.Size(570, 18);
    this.progressBar1.Style = System.Windows.Forms.ProgressBarStyle.Marquee;
    this.progressBar1.TabIndex = 1;
    // 
    // label1
    // 
    this.label1.AutoSize = true;
    this.label1.Font = new System.Drawing.Font("Calibri", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
    this.label1.Location = new System.Drawing.Point(11, 8);
    this.label1.Name = "label1";
    this.label1.Size = new System.Drawing.Size(316, 21);
    this.label1.TabIndex = 2;
    this.label1.Text = "Running dotnet trace...";
    // 
    // CustomMsgBox
    // 
    this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
    this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
    this.ClientSize = new System.Drawing.Size(595, 131);
    this.Controls.Add(this.label1);
    this.Controls.Add(this.progressBar1);
    this.Controls.Add(this.button1);
    this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
    this.MaximizeBox = false;
    this.MinimizeBox = false;
    this.Name = "CustomMsgBox";
    this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
    this.Text = "Tizen .NET Diagnostics";
    this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.CustomMsgBox_FormClosing);
    this.ResumeLayout(false);
    this.PerformLayout();

}
