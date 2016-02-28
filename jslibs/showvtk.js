//goog.require('X.renderer3D');
//goog.require('X.mesh');

window.onload = function() {

  // create and initialize three 3D renderers
  var r1 = new X.renderer3D();
  // .. attach the renderer to a <div> container using its id
  r1.container = 'C10';
  r1.init();
  r1.camera.position = [0, 0, -100];
  r1.camera.up = [0, 1, 0];
  

  // create a new X.mesh and attach a .VTK file
  var mesh = new X.mesh();
  mesh.file = 'file:///media/AMMONIS/projects/mtt_anat/controls/tractography/C10/test.vtk';
  
  // .. but add it to only to the first renderer
  r1.add(mesh);
  
  
  // start the loading of the .VTK file and display it on renderer r1.
  // once the file was fully loaded, the r1.onShowtime function is executed
  r1.render();
  
  /*
   * Thank you:
   * 
   * The rendered vessel is an arteriovenous fistula in an arm (a bypass created
   * by joining an artery and vein) acquired from a MR scanner. The data was
   * provided by Orobix S.R.L.
   * 
   * http://www.orobix.com
   */

};
