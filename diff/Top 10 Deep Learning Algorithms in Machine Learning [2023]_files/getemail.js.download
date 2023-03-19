function validateEmail(email){
	var re = /^([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?)$/i;
	return re.test(email);
}



$(document).ready(function() {

	var is_submit = 0;
	var type = null;
	var redirect_path = null;
		
	document.body.addEventListener('click', function clearInputs() {
		if(!document.body.classList.contains('modal-open')){
			$(".number-valid-message").html("");
			$(".email-valid-message").html("");
		}
	});
	
	$(".email_submit").click(function(e) {
		//alert("ok");
		type = $(this).attr("data-button");
		//console.log("type==="+type);
		var email = $("#email-"+type).val();
		var phone = $("#phone-"+type).val();
		var page_url = $(location).attr('href');
//		console.log("type==="+type);
//		console.log("email==="+email);
//		console.log("phone==="+phone);
	   	if(email.length == 0 || !(validateEmail(email))){
	   		$(".email-valid-message").html("Please enter valid email address");
	   		if(phone.toString().length<10){
	   			$(".number-valid-message").html("Please enter 10 digit number");
	   		}else{
	   			$(".number-valid-message").html("");
	   		}
			return false;
		}else if(phone.toString().length<10){
			$(".number-valid-message").html("Please enter 10 digit number");
			$(".email-valid-message").html("");
			return false;
		}
	   	else{
			$(".number-valid-message").html("");
			$(".email-valid-message").html("");
		}
	   	
	   	$(".email_submit").html('Please Wait');
	   	$(".email_submit").attr('disabled',true);
	   	
	    $.ajax({
	        type: "POST",
	        url: "/project/get-email",
	        
	        data: { 
	            email: email,
	            phone: phone,
	            type:type,
	            page_url: page_url
	        },
	        dataType: 'json',
	        success: function(data) {
	        	redirect_path = data.redirect_path;
	        	if(type=="startProject" || type == "blogAccessProject") {
	        		window.location.replace(redirect_path);
	        	}
	        	else {
	        		is_submit = 1;
	        		$( ".email_submit").hide();
	        		$( "#continue-"+type).show();
	        		$( "div.modal-body" ).html(data.message);
	        	}
	        },
	        
	        error: function(result) {
	        	//console.log(result);
	        	$( "div.modal-body" ).text( "An error occurred - Please contact care@projectpro.io" );
	        }
	    });
	    
	    
	    $("#continue-"+type).click(function(e) {
	    	//console.log("type in continue==="+type);
			//window.location.reload();
	    	window.location.replace(redirect_path);
		});
	    
	    $("#"+type).on("hide.bs.modal", function () {
			//console.log("inside hide==="+is_submit);
			if(is_submit==1) {
			    //window.location.reload();
				window.location.replace(redirect_path);
		    }
		});
	    
	    
	});

	/*
	$(".continue").click(function(e) {
			window.location.reload();
		}
	);
	
	
	$("#recipeSubscribe").on("hide.bs.modal", function () {
		console.log("inside hide==="+is_submit);
		if(is_submit==1) {
		    window.location.reload();
	    }
	});
	*/
});