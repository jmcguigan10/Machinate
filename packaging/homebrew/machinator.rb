class Machinator < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinator"
  url "https://github.com/jmcguigan10/Machinator/releases/download/v0.7.0/machinator-0.7.0.tar.gz"
  sha256 "e6c409173664bf290717b1a24c71a8e560407fa122807374a1975440d0e22505"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinator", shell_output("#{bin}/machinator --help")
  end
end
