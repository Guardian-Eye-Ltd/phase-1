import bcrypt

def hash_password(password: str) -> str:
    """
    Hash a plain text password using the modern bcrypt library.
    
    Bcrypt automatically generates a unique cryptographically secure salt,
    prepends it, and hashes the password.
    
    Args:
        password: The plain text password string.
        
    Returns:
        The hashed password string (UTF-8 decoded).
    """
    # Encode password to bytes as bcrypt operations require byte sequences
    password_bytes = password.encode("utf-8")
    
    # Generate a random salt with standard work factor cost (12 rounds)
    salt = bcrypt.gensalt(rounds=12)
    
    # Compute the Blowfish-based hash
    hashed_bytes = bcrypt.hashpw(password_bytes, salt)
    
    return hashed_bytes.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify an incoming plain text password against a stored bcrypt password hash.
    
    Args:
        plain_password: The raw password input.
        hashed_password: The stored hash retrieved from the database.
        
    Returns:
        True if password matches, False otherwise.
    """
    try:
        # Encode inputs to bytes for cryptographic comparison
        password_bytes = plain_password.encode("utf-8")
        hashed_bytes = hashed_password.encode("utf-8")
        
        # Safe constant-time comparison prevents timing attack analysis
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        # Prevent runtime crashes if database contains malformed hashes
        return False
