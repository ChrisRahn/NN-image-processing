$(document).ready(function(){
    console.log('Script Loaded OK!');

    $('.imgitem').click(async function(){

        const item = $(this).text()
        console.log(item)

        $('#in_img').attr('src', '/static/' + item)

        const response = await $.ajax('/predict', {
            data: JSON.stringify(item),
            method: 'post',
            contentType: 'application/json'       
        })

        console.log(response)
        
        $('#readout').text(response)
        
        $('#out_img').attr('src', '/static/prediction.png' + '?' + Math.random())
        
    })

    $('#refresh').click(async function(){

        console.log('Refresh clicked!')

        const response = await $.ajax('/refreshlist', {
            method: 'post',
            contentType: 'application/json'
        })

        console.log(response)

        var res_length = response.length;
        var new_list_str = '';
        for (var i=0; i<res_length; i++) {
            new_list_str += ''.concat('<li class=\'imgitem\'>', response[i], '</li>');
        }
        console.log(new_list_str);

        $('#imglist').html(new_list_str);

    })
})




