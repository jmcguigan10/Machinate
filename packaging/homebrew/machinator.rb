class Machinator < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinator"
  url "https://github.com/jmcguigan10/Machinator/releases/download/v0.4.0/machinator-0.4.0.tar.gz"
  sha256 "e698528d4d1132856f632e612a967e1358559e0a765f5374a6e5c8c10222ee1a"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinator", shell_output("#{bin}/macht --help")
  end
end
