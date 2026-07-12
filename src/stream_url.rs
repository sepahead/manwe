//! Strict parsing for credential-bearing RTSP stream URLs.

use std::net::Ipv6Addr;

/// Constant diagnostic used so validation never reflects credentials.
pub const INVALID_RTSP_URL: &str =
    "stream URLs must be valid rtsp:// or rtsps:// URLs with an explicit host";

/// Validate the subset of RTSP URLs accepted by the viewer.
pub fn validate_rtsp_url(value: &str) -> Result<(), &'static str> {
    if value.is_empty()
        || value.len() > 4_096
        || value.contains('#')
        || !has_valid_percent_encoding(value)
        || value.chars().any(|character| {
            character.is_control()
                || character.is_whitespace()
                || matches!(character, '\'' | '"' | '\\')
        })
    {
        return Err(INVALID_RTSP_URL);
    }
    let (scheme, remainder) = value.split_once("://").ok_or(INVALID_RTSP_URL)?;
    if !scheme.eq_ignore_ascii_case("rtsp") && !scheme.eq_ignore_ascii_case("rtsps") {
        return Err(INVALID_RTSP_URL);
    }
    let authority = remainder
        .split(['/', '?', '#'])
        .next()
        .filter(|authority| !authority.is_empty())
        .ok_or(INVALID_RTSP_URL)?;
    let host_port = if let Some((userinfo, host_port)) = authority.rsplit_once('@') {
        if userinfo.is_empty() || userinfo.contains('@') || host_port.is_empty() {
            return Err(INVALID_RTSP_URL);
        }
        host_port
    } else {
        authority
    };
    validate_host_port(host_port)
}

fn has_valid_percent_encoding(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if bytes
                .get(index + 1..index + 3)
                .is_none_or(|hex| !hex.iter().all(u8::is_ascii_hexdigit))
            {
                return false;
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    true
}

fn validate_host_port(host_port: &str) -> Result<(), &'static str> {
    if let Some(ipv6) = host_port.strip_prefix('[') {
        let (address, suffix) = ipv6.split_once(']').ok_or(INVALID_RTSP_URL)?;
        address.parse::<Ipv6Addr>().map_err(|_| INVALID_RTSP_URL)?;
        if suffix.is_empty() {
            return Ok(());
        }
        return validate_port(suffix.strip_prefix(':').ok_or(INVALID_RTSP_URL)?);
    }
    if host_port.contains(['[', ']']) || host_port.matches(':').count() > 1 {
        return Err(INVALID_RTSP_URL);
    }
    let (host, port) = host_port
        .rsplit_once(':')
        .map_or((host_port, None), |(host, port)| (host, Some(port)));
    if host.is_empty()
        || host.starts_with('.')
        || host.ends_with('.')
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(INVALID_RTSP_URL);
    }
    if let Some(port) = port {
        validate_port(port)?;
    }
    Ok(())
}

fn validate_port(port: &str) -> Result<(), &'static str> {
    let port = port.parse::<u16>().map_err(|_| INVALID_RTSP_URL)?;
    if port == 0 {
        Err(INVALID_RTSP_URL)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_accepts_rtsp_credentials_and_ipv6() {
        assert!(validate_rtsp_url("rtsp://user:pass@example.invalid:554/live").is_ok());
        assert!(validate_rtsp_url("rtsps://[2001:db8::1]/camera").is_ok());
    }

    #[test]
    fn validator_rejects_other_schemes_and_malformed_authorities() {
        assert_eq!(
            validate_rtsp_url("file:///etc/passwd"),
            Err(INVALID_RTSP_URL)
        );
        assert_eq!(
            validate_rtsp_url("rtsp://user:password/path"),
            Err(INVALID_RTSP_URL)
        );
        assert_eq!(
            validate_rtsp_url("rtsp://example.invalid/camera'\nfile '/tmp/other"),
            Err(INVALID_RTSP_URL)
        );
        assert_eq!(
            validate_rtsp_url("rtsp://example.invalid/live%0"),
            Err(INVALID_RTSP_URL)
        );
        assert_eq!(
            validate_rtsp_url("rtsp://example.invalid/live#fragment"),
            Err(INVALID_RTSP_URL)
        );
    }
}
