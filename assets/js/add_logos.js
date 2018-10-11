
var logo_element = '\
<div class="technion-logo" style="margin-right: 10px; width: 180px; height: auto;"> \
<a class="" href="https://cs.technion.ac.il" style="display: inline-block"> \
            <img src="/assets/images/cs_technion-logo.png" alt=""> \
        </a> \
</div> \
'

document
    .querySelector('.masthead__inner-wrap')
    .insertAdjacentHTML('afterbegin', logo_element);
