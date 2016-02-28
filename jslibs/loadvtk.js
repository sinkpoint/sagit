// goog.require('X.renderer3D');
// goog.require('X.volume');
// goog.require('X.fibers');

var XTK_ctx = {};
var VOL_ctx = {};
var GUI_ctx = {};
var OBJ_CACHE = {};

function unloadVolume(sid)
{
	var ren = XTK_ctx[sid];	
	var volume = VOL_ctx[sid];
	var gui = GUI_ctx[sid];
	if (volume != null)
	{
		ren.remove(volume);
	}
	if (gui != null)
	{
		gui.domElement.parentNode.removeChild(gui.domElement);
		GUI_ctx[sid] = null;
	}

}

function loadVolume(sid, img_type)
{
	var ren = XTK_ctx[sid];
	var volume = VOL_ctx[sid];
	if (volume == null)
	{
		volume = new X.volume();
		VOL_ctx[sid] = volume;
	}
	else {
		chkboxes = $('#'+sid+' :checkbox');
		chkboxes.each(function(){

			if ($(this).attr("name")!=img_type)
				$(this).prop("checked", false);
		});
		unloadVolume(sid);
	}

	data_array = VTK_FILES[sid];
	contain = $('#'+sid);

	subj_data = data_array[1]
	volume.file= subj_data[img_type][0];
	volume.labelmap.file = subj_data[img_type][1];
	volume.labelmap.colortable.file = 'http://x.babymri.org/?genericanatomy.txt';
	//volume.indexY = -1;
	//volume.labelmap.colortable.file = '';
	

	ren.onShowtime = function () {
		volume.indexY = false;
		var gui = GUI_ctx[sid]
		if (gui == null)
		{

			gui = new dat.GUI();

			//contain.find("div.controls").append(gui.domElement);
			contain.append(gui.domElement);
			gui.domElement.style.position='absolute';
			gui.domElement.style.top='0px';
			gui.domElement.style.right='0px';
			gui.domElement.style.overflow='visible';
			gui.domElement.style.zIndex="1000"			
			GUI_ctx[sid] = gui;
		}
		    var sliceZController = gui.add(volume, 'indexZ', 0, volume.dimensions[2] - 1).listen();			
		    var sliceXController = gui.add(volume, 'indexX', 0, volume.dimensions[0] - 1).listen();
		    var sliceYController = gui.add(volume, 'indexY', 0, volume.dimensions[1] - 1).listen();

		    var opacityController = gui.add(volume, 'opacity', 0, 1);
		    // .. and the threshold in the min..max range
		    var lowerThresholdController = gui.add(volume, 'lowerThreshold',
		        volume.min, volume.max);
		    var upperThresholdController = gui.add(volume, 'upperThreshold',
		        volume.min, volume.max);
		    var lowerWindowController = gui.add(volume, 'windowLow', volume.min,
		        volume.max);
		    var upperWindowController = gui.add(volume, 'windowHigh', volume.min,
		        volume.max);		    

		    volume.indexX = 0;
		    volume.indexY = 0;
		    gui.open();					

	};	

	ren.add(volume);


}

function loadVtk(contain, data_array)
{

	//console.log(contain);
	//console.log(data_array);

	var id = contain.attr('id');
	var ren = XTK_ctx[id]


	if (ren == null)	
	{

	    ren = new X.renderer3D();
		ren.container = id;
		ren.init();
		ren.camera.position = [0,0,-100];
		ren.camera.up = [0,1,0];

		fib_file = data_array[0];
		var mesh = OBJ_CACHE[fib_file];
		if (mesh == null )
		{
			mesh = new X.fibers();
			mesh.file = fib_file;
			OBJ_CACHE[fib_file];
		}
		mesh.color = [1,0.25,0.25];
		mesh.linewidth = 1;
		ren.add(mesh);

		// var volume = new X.volume();
		// volume.file= data_array[1];
		// volume.labelmap.file = data_array[2];
		// volume.labelmap.colortable.file = 'http://x.babymri.org/?genericanatomy.txt';
		// //volume.indexY = -1;
		// //volume.labelmap.colortable.file = '';

		// ren.add(volume);

		// ren.onShowtime = function () {
		// 	volume.indexY = false;
		// 	var gui = GUI_ctx[id]
		// 	if (gui == null)
		// 	{

		// 		gui = new dat.GUI();

		// 		gui.domElement.style.positon='absolute';
		// 		gui.domElement.style.top='0px';
		// 		gui.domElement.style.left='0px';
		// 		gui.domElement.style.zIndex="1000"
		// 		contain.find("div.controls").append(gui.domElement);
		// 		GUI_ctx[id] = gui;
		// 	}
		// 	    var sliceZController = gui.add(volume, 'indexZ', 0, volume.dimensions[2] - 1).listen();			
		// 	    var sliceXController = gui.add(volume, 'indexX', 0, volume.dimensions[0] - 1).listen();
		// 	    var sliceYController = gui.add(volume, 'indexY', 0, volume.dimensions[1] - 1).listen();

		// 	    volume.indexX = 0;
		// 	    volume.indexY = 0;
		// 	    gui.open();					

		// };

		ren.render();
		XTK_ctx[id] = ren;

		return ren;
	}
}

function unloadVtk(contain)
{
	var id = contain.attr('id');	
	var ren = XTK_ctx[id];
	var gui = GUI_ctx[id];
	var volume = VOL_ctx[id];

	if (gui != null)
	{
		gui.domElement.parentNode.removeChild(gui.domElement);
		GUI_ctx[id] = null;		
	}

	if (ren != null)
	{
		// if (volume != null)	
		// {
		// 	ren.remove(volume);
		// 	VOL_ctx[id] = null;
		// }		
		ren.destroy();
		XTK_ctx[id] = null;

	}



}

function toggleFullScreen(id, btn)
{

	var elt = $('#'+id);

	if (elt.css('position') == 'relative') {
		btn.attr('value', 'Shrink');
		elt.css('position','absolute');
		elt.css('width','100%');
		elt.css('height','100%');
		elt.css('left','0');
		elt.css('top',$(window).scrollTop());
		elt.css('z-index','100000');		

		var elt = $('#'+id+' canvas');
		elt.attr('{width:'+$(window).innerWidth()+',height:'+$(window).innerHeight()+'}')
	}
	else {
		btn.attr('value', 'Expand');
		elt.css('position','relative');
		elt.css('width','49.9%');
		elt.css('height','600px');
		elt.css('top','0');
		elt.css('z-index','0');

		var elt = $('#'+id+' canvas');
		elt.attr('{width:"100%",height:"100%"}')		
	}

	var ren = XTK_ctx[id];
	var evt = document.createEvent('UIEvents');
	evt.initUIEvent('resize', true, false,window,0);
	window.dispatchEvent(evt);
}
