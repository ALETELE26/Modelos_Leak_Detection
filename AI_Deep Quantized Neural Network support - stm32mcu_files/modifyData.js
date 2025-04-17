// isIframe
if( window.location !== window.parent.location ) {
	digitalData.page.pageInfo.isIframe = 'Yes';
}

//contentIframe
if( window.parent.frames.length > 0 ) {
	digitalData.page.pageInfo.contentIframe = 'Yes';
}

var referrerUrl = document.referrer;

digitalData.page.pageInfo.referringURL = referrerUrl;

if( referrerUrl ) {
	var pageName = getPageNameFromUrl( referrerUrl );
	if( pageName ) {
		digitalData.page.pageInfo.previous.pageShortName = pageName;
	}
}

if( isMobileByUserAgent() && isMobileByOrientationAbility() ) {
	digitalData.page.pageInfo.sysEnv = 'mobile';
}

// Best effort to get page name
function getPageNameFromUrl( url ) {
	var re = /(index.php\/|title=)([\s\S][^\&\?]*)/;
	var matches = re.exec( url );
	if( matches && matches.length > 0 ) {
		return matches[2];
	}
	return '';
}

// UserAgent is quite reliable, but might not detect
// rare devices
function isMobileByUserAgent() {
	if( navigator.userAgent.match(/Android/i)
		|| navigator.userAgent.match(/webOS/i)
		|| navigator.userAgent.match(/iPhone/i)
		|| navigator.userAgent.match(/iPad/i)
		|| navigator.userAgent.match(/iPod/i)
		|| navigator.userAgent.match(/BlackBerry/i)
		|| navigator.userAgent.match(/Windows Phone/i)
	){
		return true;
	}
	else {
		return false;
	}
}

// Usually only mobile devices have orientation set,
// but its not for Windows PCs to have it as well
function isMobileByOrientationAbility() {
	if (typeof window.orientation !== 'undefined') {
		return true;
	}
	return false;
}
