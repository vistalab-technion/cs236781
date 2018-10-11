
var logo_element = '\
<div class="technion-logo" style="margin-right: 10px; width: 180px; height: auto;"> \
<a class="" href="https://cs.technion.ac.il" style="display: inline-block"> \
            <img src="/assets/images/cs_technion-logo.png" alt="Technion"> \
        </a> \
</div> \
';

document
    .querySelector('.masthead__inner-wrap')
    .insertAdjacentHTML('afterbegin', logo_element);


var logo_element = '\
<div class="vista-logo" style="float: inline-end; margin: 1em"> \
<a class="" style="display: block; margin: 0 auto; width: 250px" href="https://vista.cs.technion.ac.il" > \
            <img src="/assets/images/vista-logo-bw.png" alt="VISTA"> \
        </a> \
</div> \
';

var footerNodes = document.getElementsByTagName("FOOTER")
var footerNode = footerNodes[footerNodes.length - 1];
footerNode.style.display = 'inline-block';
footerNode.insertAdjacentHTML('afterend', logo_element);

